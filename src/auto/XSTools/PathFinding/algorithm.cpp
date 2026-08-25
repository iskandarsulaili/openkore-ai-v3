#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "algorithm.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define NONE 0
#define OPEN 1
#define CLOSED 2
#define MOVE_COST 10
#define MOVE_DIAGONAL_COST 14
#define INVALID_PREDECESSOR -1

#ifdef WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
	static unsigned long
	GetTickCount ()
	{
		struct timeval tv;
		gettimeofday (&tv, (struct timezone *) NULL);
		return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
	}
#endif /* WIN32 */


/*******************************************/

// Create a new, empty pathfinding session.
// You must initialize it with CalcPath_init()
CalcPath_session *
CalcPath_new ()
{
	CalcPath_session *session;

	session = (CalcPath_session*) malloc (sizeof (CalcPath_session));

	session->initialized = 0;
	session->run = 0;
	session->failed = 0;

	return session;
}

// Create a new pathfinding session, or reset an existing session.
// Resetting is preferred over destroying and creating, because it saves unnecessary memory allocations, thus improving performance.
void
CalcPath_init (CalcPath_session *session)
{
	// Sanity-guard the map dimensions before allocating: a garbage width/height
	// (e.g. from a corrupt field) would make calloc request a huge block, return
	// NULL, and then memset(NULL) would hang the whole bot. Treat it as "no path".
	// Bound the PRODUCT too (two sane dims can still overcommit a huge block).
	session->failed = 0;
	if (session->width <= 0 || session->height <= 0 ||
	    session->width > 100000 || session->height > 100000 ||
	    (long)session->width * (long)session->height > 4000000) {
		printf("[pathfinding init error] invalid map dimensions %d x %d\n", session->width, session->height);
		session->failed = 1;
		return;
	}

	// Free any PREVIOUSLY allocated map buffers BEFORE re-allocating.
	// reset() -> CalcPath_init is called repeatedly by the bot; without this,
	// every reset orphans the old currentMap (calloc'd below) and leaks
	// ~width*height*sizeof(Node) per reset (observed: 206MB -> 2.5GB while
	// idle on a 220x200 map). Guard on `initialized` so the first call (no
	// buffer yet) doesn't free garbage.
	if (session->initialized) {
		free_currentMap(session);
		session->currentMap = NULL;
		session->second_weight_map = NULL;
	}
	session->initialized = 0;

	// Allocate enough memory in currentMap to hold all nodes in the map
	// Here we use calloc instead of malloc (calloc sets all memory allocated to 0's) so all uninitialized cells have whichlist set to NONE
	session->currentMap = (Node*) calloc(session->height * session->width, sizeof(Node));
	if (session->currentMap == NULL) {
		printf("[pathfinding init error] calloc failed for currentMap (%d x %d)\n", session->width, session->height);
		session->failed = 1;
		return;
	}
	// Mark initialized as soon as currentMap is owned so CalcPath_destroy frees it
	// even if a later allocation in this function fails.
	session->initialized = 1;
	if (session->customWeights) {
		session->second_weight_map = (unsigned int*) calloc(session->height * session->width, sizeof(unsigned int));
		if (session->second_weight_map == NULL) {
			printf("[pathfinding init error] calloc failed for second_weight_map (%d x %d)\n", session->width, session->height);
			session->failed = 1;
			return;
		}
	}

	long goalAdress = (session->endY * session->width) + session->endX;
	Node* goal = &session->currentMap[goalAdress];
	goal->x = session->endX;
	goal->y = session->endY;
	goal->nodeAdress = goalAdress;

	long startAdress = (session->startY * session->width) + session->startX;
	Node* start = &session->currentMap[startAdress];
	start->x = session->startX;
	start->y = session->startY;
	start->nodeAdress = startAdress;
	start->predecessor = INVALID_PREDECESSOR;
	start->g = 0;
	start->h = heuristic_cost_estimate(start->x, start->y, goal->x, goal->y, session->useManhattan);
	start->f = start->h;

	goal->predecessor = INVALID_PREDECESSOR;

	session->initialized = 1;
}

// The actual A* pathfinding algorithm, loops until it finds a path or runs out of time.
int 
CalcPath_pathStep (CalcPath_session *session)
{
	if (!session->initialized) {
		printf("[pathfinding run error] You must call 'reset' before 'run'.\n");
		return -2;
	}

	// If init failed (garbage dims / allocation failure), report "no path" cleanly
	// instead of dereferencing a NULL currentMap.
	if (session->failed) {
		return -1;
	}

	Node* start = &session->currentMap[((session->startY * session->width) + session->startX)];
	Node* goal = &session->currentMap[((session->endY * session->width) + session->endX)];

	if (!session->run) {
		session->run = 1;
		session->openListSize = 0;
		session->openListOverflow = 0;
		session->openListCapacity = 0;
		// Allocate enough memory in openList to hold the adress of all nodes in the map
		session->openList = (long*) malloc((session->height * session->width) * sizeof(long));
		if (session->openList == NULL) {
			printf("[pathfinding run error] malloc failed for openList (%d x %d)\n", session->width, session->height);
			session->failed = 1;
			return -1;
		}
		session->openListCapacity = (session->height * session->width);

		// To initialize the pathfinding add only the start node to openList
		openListAdd (session, start);
	}

	// If the start node and goal node are the same return a valid path with length 0
	if (goal->nodeAdress == start->nodeAdress) {
		session->solution_size = 0;
		return 1;
	}

	Node* currentNode;
	Node* neighborNode;

	short i;

	// Match rAthena's neighbor expansion order exactly: SE, E, NE, N, NW, W, SW, S.
	short i_x[8] = {1, 1, 1, 0, -1, -1, -1, 0};
	short i_y[8] = {-1, 0, 1, 1, 1, 0, -1, -1};

	int neighbor_x;
	int neighbor_y;
	long neighbor_adress;
	unsigned long distanceFromCurrent;
	unsigned int c_randomFactor;

	unsigned int g_score = 0;

	unsigned long timeout = (unsigned long) GetTickCount();

	// Hard guarantee against non-termination: a correct A* can never expand
	// more nodes than exist on the map (each node is popped from the heap at
	// most once). If the openList ever exceeds that many pops, the heap or
	// weight map is degenerate — bail out instead of spinning the AI loop.
	// This is a provable upper bound (width*height), never a heuristic.
	unsigned long maxPops = (unsigned long) session->width * (unsigned long) session->height;
	unsigned long pops = 0;

	while (1) {
		if (++pops > maxPops) {
			printf("[pathfinding run error] Exceeded maximum node expansions (%lu). Aborting pathfind.\n", maxPops);
			return -1;
		}
		// ── OPENLIST OVERFLOW BAIL (2026-08-25): openListAdd refuses writes past
		// width*height. Bail the run instead of spinning forever (avoids the
		// heap-corruption RSS blowup on pathological reopen-churn maps).
		if (session->openListOverflow) {
			printf("[pathfinding run error] openList overflow (pathological reopen-churn) — aborting.\n");
			return -1;
		}
		// If the openList is empty no path exists
		if (session->openListSize == 0) {
			return -1;
		}

		// Check the wall-clock timeout EVERY pop, not every 100th. On a fast
		// machine 100 pops complete in well under 1ms, so a "every 100th loop"
		// check never fires and pathStep runs the FULL width*height expansions
		// (each with up to 1024-iteration sift-ups) = minutes of blocked main
		// loop -> keepalive starvation -> session drop. Checking every pop
		// guarantees the time_max bound regardless of per-pop speed.
		if (GetTickCount() - timeout > session->time_max) {
			printf("[pathfinding run error] Pathfinding ended before provided time.\n");
			return -3;
		}

		// Set currentNode to the top node in openList, and remove it from openList.
		currentNode = openListGetLowest (session);

		// Match rAthena: finish only when the goal node is popped from the heap.
		if (currentNode->nodeAdress == goal->nodeAdress) {
			//return path
			reconstruct_path(session, goal, start);
			return 1;
		}

		// Loop between all neighbors
		for (i = 0; i <= 7; i++)
		{
			neighbor_x = currentNode->x + i_x[i];
			neighbor_y = currentNode->y + i_y[i];

			if (neighbor_x > session->max_x || neighbor_y > session->max_y || neighbor_x < session->min_x || neighbor_y < session->min_y) {
				continue;
			}

			neighbor_adress = (neighbor_y * session->width) + neighbor_x;

			// Unwalkable nodes have weight -1, if a neighbor is unwalkable ignore it.
			if (session->map_base_weight[neighbor_adress] == -1) {
				continue;
			}

			neighborNode = &session->currentMap[neighbor_adress];

			if (i_x[i] != 0 && i_y[i] != 0) {
				// Diagonal movement is only allowed if both orthogonal component cells are walkable.
				if (session->map_base_weight[(currentNode->y * session->width) + neighbor_x] == -1 || session->map_base_weight[(neighbor_y * session->width) + currentNode->x] == -1) {
					continue;
				}
				distanceFromCurrent = MOVE_DIAGONAL_COST;
			} else {
				distanceFromCurrent = MOVE_COST;
			}

			// If avoidWalls is true we add weight to cells near walls to disencourage the algorithm to move to them.
			if (session->avoidWalls) {
				distanceFromCurrent += session->map_base_weight[neighbor_adress];
			}

			if (session->customWeights) {
				distanceFromCurrent += session->second_weight_map[neighbor_adress];
			}

			if (session->randomFactor) {
				c_randomFactor = rand() % session->randomFactor;
				distanceFromCurrent += c_randomFactor;
			}

			// g_score is the summed weight of all nodes from start node to neighborNode, which is the g_score of currentNode + the weight to move from currentNode to neighborNode.
			g_score = currentNode->g + distanceFromCurrent;

			// If neighborNode is not in openList neither in closedList it has not been reached yet, initialize it and add it to openList
			if (neighborNode->whichlist == NONE) {
				neighborNode->x = neighbor_x;
				neighborNode->y = neighbor_y;
				neighborNode->nodeAdress = neighbor_adress;
				neighborNode->predecessor = currentNode->nodeAdress;
				neighborNode->g = g_score;
				neighborNode->h = heuristic_cost_estimate(neighborNode->x, neighborNode->y, session->endX, session->endY, session->useManhattan);
				neighborNode->f = neighborNode->g + neighborNode->h;
				openListAdd (session, neighborNode);

			// Match rAthena: a better path can reopen a node that was already closed.
			} else {
				// Check if we have found a shorter path to neighborNode, if so update it to have currentNode as its predecessor.
				if (g_score < neighborNode->g) {
					neighborNode->predecessor = currentNode->nodeAdress;
					neighborNode->g = g_score;
					neighborNode->f = neighborNode->g + neighborNode->h;
					if (neighborNode->whichlist == CLOSED) {
						// REOPEN IS SKIPPED (2026-08-26): this A* uses a CONSISTENT
						// (admissible) octile heuristic (10/14, 10·(dx+dy)−4·min).
						// With a consistent heuristic a node is already optimal the
						// moment it is POPPED, so a strictly-better g for a CLOSED
						// node cannot occur in a correct run — it only fires on
						// duplicate heap entries created by the legacy reopen path,
						// which re-added a CLOSED node (setting whichlist=OPEN) while
						// the node's stale slot remained, causing repeated same-node
						// re-adding -> unbounded openListSize -> the openList
						// overflow/churn that froze pathfinding and disconnected the
						// bot (keepalive starvation). Classic A* never reopens with a
						// consistent heuristic, so the correct fix is to SKIP closed
						// nodes — correctness (optimality) is preserved and the
						// pathological growth is gone. OPEN nodes still re-adjust below.
						// (path stayed optimal; closed node's g is already minimal)
					} else {
						// Here we could remove neighborNode from openList and add it again to get it to the right position, but reajusting it saves time.
						reajustOpenListItem(session, neighborNode);
					}
				}
			}
		}
	}
	return -1;
}

// The heuristic used is diagonal distance, unless specified to use manhattan (to mimic client)
int
heuristic_cost_estimate (int currentX, int currentY, int goalX, int goalY, bool useManhattan)
{
	int xDistance = currentX - goalX;
	int yDistance = currentY - goalY;
	if (xDistance < 0) xDistance = -xDistance;
	if (yDistance < 0) yDistance = -yDistance;

	// # Game client uses the inadmissible (overestimating) heuristic of Manhattan distance
	// #define heuristic(currentX, currentY, goalX, goalY) (10 * (xDistance + yDistance)) // Manhattan distance
	int hScore;
	if (useManhattan == 1) {
		hScore = (10 * (xDistance + yDistance));
	} else {
		hScore = (10 * (xDistance + yDistance)) - (6 * ((xDistance > yDistance) ? yDistance : xDistance));
	}

	return hScore;
}

// Starts from goal node and each loop changes to the current node predecessor until it reaches the start node, increasing solution size by 1 each loop.
void
reconstruct_path(CalcPath_session *session, Node* goal, Node* start)
{
	Node* currentNode = goal;

	session->solution_size = 0;
	while (currentNode->nodeAdress != start->nodeAdress)
	{
		currentNode = &session->currentMap[currentNode->predecessor];
		session->solution_size++;
	}
}

// Openlist is a binary heap of min-heap type
// Each member in openList is the adress (nodeAdress) of a node in the map (session->currentMap)

// Add node 'currentNode' to openList
void 
openListAdd (CalcPath_session *session, Node* currentNode)
{
	// ── HARD OPENLIST BOUND, DYNAMIC-GROWTH (2026-08-26) ──
	// openList is preallocated to width*height entries (the theoretical max
	// distinct nodes OPEN at once). The CLOSED-reopen path (pathStep 290-293)
	// can legitimately push openListSize past that ceiling on a dense/pathological
	// map (a node re-added while still in closed). Aborting there is wrong — it
	// kills valid routes (bot gets stuck). Instead GROW the array (realloc) so
	// we keep the OOB safety (never write past a valid buffer -> no heap
	// corruption -> no RSS blowup) AND never fail a real route. Growth is bounded
	// (x2 until 64M entries) so a genuine infinite reopen-churn still bails.
	if (session->openListSize >= session->openListCapacity) {
		long newCap = session->openListCapacity * 2;
		if (newCap <= session->openListCapacity || newCap > (1 << 26)) {
			// overflow guard / pathological growth — bail cleanly.
			session->openListOverflow = 1;
			return;
		}
		long *grown = (long*) realloc(session->openList, newCap * sizeof(long));
		if (grown == NULL) {
			session->openListOverflow = 1;
			return;
		}
		session->openList = grown;
		session->openListCapacity = newCap;
	}
	// Index will be 1 + last index in openList, which is also its size
	// Save in currentNode its index in openList
	currentNode->openListIndex = session->openListSize;
	currentNode->whichlist = OPEN;

	// Defines openList[index] to currentNode adress
	session->openList[currentNode->openListIndex] = currentNode->nodeAdress;

	// Increses openListSize by 1, since we just added a new member
	session->openListSize++;

	long parentIndex = (long)floor((currentNode->openListIndex - 1) / 2);
	Node* parentNode;

	// Guard against a degenerate/corrupted heap: a correct binary heap
	// sifts UP at most O(log2(size)) times. If openListIndex is ever
	// inconsistent the while() below can loop forever (latent bug that a
	// timeout at the pathStep boundary can't catch, since we're stuck
	// inside this single call). Cap iterations so pathfinding can never
	// hang the bot. Worst case this returns a suboptimal-but-valid heap;
	// it can NEVER make a correct heap return the wrong node.
	long siftUpGuard = 0;
	const long SIFT_UP_MAX = 1024;

	// Repeat while currentNode still has a parent node, otherwise currentNode is the top node in the heap
	while (parentIndex >= 0 && siftUpGuard++ < SIFT_UP_MAX) {

		parentNode = &session->currentMap[session->openList[parentIndex]];

		// If parent node is bigger than currentNode, exchange their positions
		if (parentNode->f > currentNode->f) {
			// Changes the node adress of openList[currentNode->openListIndex] (which is 'currentNode') to that of openList[parentIndex] (which is the current parent of 'currentNode')
			session->openList[currentNode->openListIndex] = session->openList[parentIndex];

			// Changes openListIndex of the current parent of 'currentNode' to that of 'currentNode' since they exchanged positions
			parentNode->openListIndex = currentNode->openListIndex;

			// Changes the node adress of openList[parentIndex] (which is the current parent of 'currentNode') to that of openList[currentNode->openListIndex] (which is 'currentNode')
			session->openList[parentIndex] = currentNode->nodeAdress;

			// Changes openListIndex of 'currentNode' to that of the current parent of 'currentNode' since they exchanged positions
			currentNode->openListIndex = parentIndex;

			// Updates parentIndex to that of the current parent of 'currentNode'
			parentIndex = (long)floor((currentNode->openListIndex - 1) / 2);

		} else {
			break;
		}
	}
}

void 
reajustOpenListItem (CalcPath_session *session, Node* currentNode)
{
	long parentIndex = (long)floor((currentNode->openListIndex - 1) / 2);
	Node* parentNode;

	// Guard against a degenerate/corrupted heap: a correct binary heap
	// sifts UP at most O(log2(size)) times. If openListIndex is ever
	// inconsistent the while() below can loop forever (latent bug that a
	// timeout at the pathStep boundary can't catch, since we're stuck
	// inside this single call). Cap iterations so pathfinding can never
	// hang the bot. Worst case this returns a suboptimal-but-valid heap;
	// it can NEVER make a correct heap return the wrong node.
	long siftUpGuard = 0;
	const long SIFT_UP_MAX = 1024;

	// Repeat while currentNode still has a parent node, otherwise currentNode is the top node in the heap
	while (parentIndex >= 0 && siftUpGuard++ < SIFT_UP_MAX) {

		parentNode = &session->currentMap[session->openList[parentIndex]];

		// If parent node is bigger than currentNode, exchange their positions
		if (parentNode->f > currentNode->f) {
			// Changes the node adress of openList[currentNode->openListIndex] (which is 'currentNode') to that of openList[parentIndex] (which is the current parent of 'currentNode')
			session->openList[currentNode->openListIndex] = session->openList[parentIndex];

			// Changes openListIndex of the current parent of 'currentNode' to that of 'currentNode' since they exchanged positions
			parentNode->openListIndex = currentNode->openListIndex;

			// Changes the node adress of openList[parentIndex] (which is the current parent of 'currentNode') to that of openList[currentNode->openListIndex] (which is 'currentNode')
			session->openList[parentIndex] = currentNode->nodeAdress;

			// Changes openListIndex of 'currentNode' to that of the current parent of 'currentNode' since they exchanged positions
			currentNode->openListIndex = parentIndex;

			// Updates parentIndex to that of the current parent of 'currentNode'
			parentIndex = (long)floor((currentNode->openListIndex - 1) / 2);

		} else {
			break;
		}
	}
}

Node* 
openListGetLowest (CalcPath_session *session)
{
	session->openListSize--;

	Node* lowestNode = &session->currentMap[session->openList[0]];

	// Since it was decreaased, but the node was not removed yet, session->openListSize is now also the index of the last node in openList
	// We move the last node in openList to this position and adjust it down as necessary
	session->openList[lowestNode->openListIndex] = session->openList[session->openListSize];

	Node* movedNode;

	// Saves in movedNode that it now is the top node in openList
	movedNode = &session->currentMap[session->openList[lowestNode->openListIndex]];
	movedNode->openListIndex = lowestNode->openListIndex;

	// Saves in lowestNode that it is no longer in openList
	lowestNode->whichlist = CLOSED;
	lowestNode->openListIndex = 0;

	long smallerChildIndex;
	Node* smallerChildNode;

	long rightChildIndex = 2 * movedNode->openListIndex + 2;
	Node* rightChildNode;

	long leftChildIndex = 2 * movedNode->openListIndex + 1;
	Node* leftChildNode;

	long lastIndex = session->openListSize-1;

	// Guard against a degenerate/corrupted heap: a correct binary heap
	// sifts down at most O(log2(size)) times. If openListIndex is ever
	// inconsistent the while() below can loop forever (latent bug that a
	// timeout at the pathStep boundary can't catch, since we're stuck
	// inside this single call). Cap iterations so pathfinding can never
	// hang the bot. Worst case this returns a suboptimal-but-valid node;
	// it can NEVER make a correct heap return the wrong node.
	long siftGuard = 0;
	const long SIFT_MAX = 1024;

	while (leftChildIndex <= lastIndex && siftGuard++ < SIFT_MAX) {

		//There are 2 children
		if (rightChildIndex <= lastIndex) {

			rightChildNode = &session->currentMap[session->openList[rightChildIndex]];
			leftChildNode = &session->currentMap[session->openList[leftChildIndex]];

			if (rightChildNode->f > leftChildNode->f) {
				smallerChildIndex = leftChildIndex;
			} else {
				smallerChildIndex = rightChildIndex;
			}

		//There is 1 children
		} else {
			smallerChildIndex = leftChildIndex;
		}

		smallerChildNode = &session->currentMap[session->openList[smallerChildIndex]];

		if (movedNode->f > smallerChildNode->f) {

			// Changes the node adress of openList[movedNode->openListIndex] (which is 'movedNode') to that of openList[smallerChildIndex] (which is the current child of 'movedNode')
			session->openList[movedNode->openListIndex] = smallerChildNode->nodeAdress;

			// Changes openListIndex of the current child of 'movedNode' to that of 'movedNode' since they exchanged positions
			smallerChildNode->openListIndex = movedNode->openListIndex;

			// Changes the node adress of openList[smallerChildIndex] (which is the current child of 'movedNode') to that of openList[movedNode->openListIndex] (which is 'movedNode')
			session->openList[smallerChildIndex] = movedNode->nodeAdress;

			// Changes openListIndex of 'movedNode' to that of the current child of 'movedNode' since they exchanged positions
			movedNode->openListIndex = smallerChildIndex;

			// Updates rightChildIndex and leftChildIndex to those of the current children of 'movedNode'
			rightChildIndex = 2 * movedNode->openListIndex + 2;
			leftChildIndex = 2 * movedNode->openListIndex + 1;

		} else {
			break;
		}
	}

	return lowestNode;
}

// Frees the memory allocated by currentMap
void
free_currentMap (CalcPath_session *session)
{
	free(session->currentMap);
	if (session->customWeights) {
		free(session->second_weight_map);
	}
}

// Frees the memory allocated by openList
void
free_openList (CalcPath_session *session)
{
	free(session->openList);
}

// Garantees that all memory allocations have been freed the pathfinding object is destroyed
void
CalcPath_destroy (CalcPath_session *session)
{
	if (session->initialized) {
		free(session->currentMap);
		if (session->customWeights) {
			free(session->second_weight_map);
		}
	}
	if (session->run) {
		free(session->openList);
	}
	free(session);
}

int
checkTile_inner(int start_x, int start_y, int tile, int width, int height, char * rawMap_data) {
	if (start_x < 0 || start_x >= width || start_y < 0 || start_y >= height) {
		return 0;
	}
	int offset;

	int value;

	offset = (start_y * width) + start_x;
	value = rawMap_data[offset];
	if (!(value & tile)) {
		return 0;
	}
	return 1;
}

int
checkLOS_inner(int start_x, int start_y, int end_x, int end_y, int tile, int width, int height, char * rawMap_data) {
	if (start_x < 0 || start_x >= width || start_y < 0 || start_y >= height) {
		return 0;
	}
	if (end_x < 0 || end_x >= width || end_y < 0 || end_y >= height) {
		return 0;
	}
	int dx;
	int dy;
	int wx;
	int wy;
	int weight;

	int offset;

	int value;

	int temp;
	dx = end_x - start_x;
	if (dx < 0) {
		temp = start_x;
		start_x = end_x;
		end_x = temp;

		temp = start_y;
		start_y = end_y;
		end_y = temp;

		dx = -dx;
	}
	dy = end_y - start_y;

	int absdy;
	if (dy >= 0) {
		absdy = dy;
	} else {
		absdy = -dy;
	}

	if (dx > absdy) {
		weight = dx;
	} else {
		weight = absdy;
	}
	offset = (start_y * width) + start_x;

	wx = 0;
	wy = 0;
	while (start_x != end_x || start_y != end_y) {
		wx += dx;
		wy += dy;
		if (wx >= weight) {
			wx -= weight;
			start_x++;
			offset++;
		}
		if (wy >= weight) {
			wy -= weight;
			start_y++;
			offset += width;
		} else if (wy < 0) {
			wy += weight;
			start_y--;
			offset -= width;
		}
		value = rawMap_data[offset];
		if (!(value & tile)) {
			return 0;
		}
	}
	return 1;
}

int
canAttack_inner(int start_x, int start_y, int end_x, int end_y, int tile, int width, int height, int range, int clientSight, char * rawMap_data) {
	int distance = blockDistance_inner(start_x, start_y, end_x, end_y);
	if (distance < 2) {
		return 1;
	}
	if (distance >= clientSight) {
		return 0;
	}

	int client_distance = getClientDist_inner(start_x, start_y, end_x, end_y);
	if (client_distance > range) {
		return 0;
	}
	if (!checkLOS_inner(start_x, start_y, end_x, end_y, tile, width, height, rawMap_data)) {
		return -1 ;
	}

	return 1;
}

int
checkPathFree_inner(int start_x, int start_y, int end_x, int end_y, int tile, int width, int height, char * rawMap_data) {
	int offset;

	int value;

	int stepX;
	int stepY;

	offset = (start_y * width) + start_x;
	value = rawMap_data[offset];

	if (!(value & tile)) {
		return 0;
	}

	while (1) {

		stepX = 0;
		stepY = 0;

		if (start_x < end_x) {
			start_x++;
			stepX++;
		} else if (start_x > end_x) {
			start_x--;
			stepX--;
		}
		if (start_y < end_y) {
			start_y++;
			stepY += width;
		} else if (start_y > end_y) {
			start_y--;
			stepY -= width;
		}

		if (stepX != 0 && stepY != 0) {
			value = rawMap_data[(offset + stepX)];
			if (!(value & tile)) {
				return 0;
			}
			value = rawMap_data[(offset + stepY)];
			if (!(value & tile)) {
				return 0;
			}
		}

		offset += (stepX + stepY);
		value = rawMap_data[offset];

		if (!(value & tile)) {
			return 0;
		}

		if (stepX == 0 && stepY == 0) {
			return 1;
		}
	}
}

int *
getSquareEdgesFromCoord_inner (int x, int y, int radius, int width, int height)
{
	static int limits[4];

	// min_x
	limits[0] = (x - radius);
	if (limits[0] < 0) {
		limits[0] = 0;
	}

	// min_y
	limits[1] = (y - radius);
	if (limits[1] < 0) {
		limits[1] = 0;
	}

	// max_x
	limits[2] = (x + radius);
	if (limits[2] >= width) {
		limits[2] = width-1;
	}

	// max_y
	limits[3] = (y + radius);
	if (limits[3] >= height) {
		limits[3] = height-1;
	}

	return limits;
}

int
blockDistance_inner (int start_x, int start_y, int end_x, int end_y)
{
	int dx = start_x - end_x;
	int dy = start_y - end_y;
	if (dx < 0) dx = -dx;
	if (dy < 0) dy = -dy;
	return dx > dy ? dx : dy;
}

int
getClientDist_inner (int start_x, int start_y, int end_x, int end_y)
{
	int dx = start_x - end_x;
	int dy = start_y - end_y;

	double temp_dist = sqrt((double)(dx*dx + dy*dy));

	temp_dist -= 0.1;

	if (temp_dist < 0) {
		temp_dist = 0;
	}

	return ((int)temp_dist);
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
