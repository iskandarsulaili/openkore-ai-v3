#!/usr/bin/env python3
"""Fix broken characters via direct rAthena protocol.

Sends raw packets to:
1. Log in to account server
2. Connect to char server  
3. Delete broken (empty-name) characters
4. Create proper characters
"""

import socket
import struct
import time
from Cryptodome.Cipher import AES

SERVER = 'asgardsglory.ddns.net'
ACC_PORT = 6900
CHAR_PORT = 6121
MAP_PORT = 5121
PASSWORD = 'b0tTib0tTi'

# Rijndael key (from OpenKore)
KEY = bytes([6, 169, 33, 64, 54, 184, 161, 91, 81, 46, 3, 213, 52, 18, 0, 6, 61, 175, 186, 66, 157, 158, 180, 48])
CHAIN = bytes([61, 175, 186, 66, 157, 158, 180, 48, 180, 34, 218, 128, 44, 159, 172, 65, 1, 2, 4, 8, 16, 32, 128])


def encrypt_password(password):
    """AES-192-ECB encryption matching OpenKore's Utils::Rijndael."""
    # Pad password to 24 bytes with nulls
    padded = password.encode('ascii').ljust(24, b'\x00')[:24]
    cipher = Cipher(algorithms.AES(KEY), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def recv_all(sock, timeout=3):
    """Receive all available data from socket."""
    buf = b''
    sock.settimeout(timeout)
    while True:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf += chunk
        except socket.timeout:
            break
    return buf


def fix_account(username):
    print(f"\n{'='*50}")
    print(f"  Fixing: {username}")
    print(f"{'='*50}")
    
    # Step 1: Connect to account server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect((SERVER, ACC_PORT))
    except Exception as e:
        print(f"  ❌ Account server: {e}")
        return
    print("  ✅ Account server connected")
    
    # Send login packet (0x0064)
    # Format: V(version=128), Z24(username), Z24(password_encrypted), C(master_version=0)
    enc_pass = encrypt_password(PASSWORD)
    login_pkt = struct.pack('<H', 0x0064)  # packet ID
    login_pkt += struct.pack('<I', 128)    # version
    login_pkt += username.encode('ascii').ljust(24, b'\x00')[:24]  # username
    login_pkt += enc_pass                   # encrypted password (24 bytes)
    login_pkt += struct.pack('B', 0)        # master_version
    
    sock.sendall(login_pkt)
    print("  📤 Sent login packet (0x0064)")
    
    time.sleep(1)
    resp = recv_all(sock, 3)
    sock.close()
    
    if len(resp) < 6:
        print(f"  ❌ Short response: {len(resp)} bytes")
        return
    
    pkt_id = struct.unpack('<H', resp[0:2])[0]
    print(f"  📦 Response packet: 0x{pkt_id:04X} ({len(resp)} bytes)")
    
    if pkt_id == 0x006A:
        err_code = struct.unpack('<H', resp[2:4])[0]
        errors = {0:'incorrect password', 1:'server closed', 2:'no account', 3:'online',
                  4:'suspended', 5:'overpopulated', 6:'maintenance', 7:'expired/GM',
                  8:'refused', 9:'rejected', 10:'too many', 11:'email', 12:'banned'}
        print(f"  ❌ Login failed: {errors.get(err_code, 'unknown')} ({err_code})")
        return
    
    if pkt_id != 0x0069:
        print(f"  ❌ Unexpected response: 0x{pkt_id:04X}")
        return
    
    # Parse account info: a4(aid), a4(sid), a4(sid2), v(ulv), C(sex)
    aid = resp[2:6]
    sid = resp[6:10]
    sid2 = resp[10:14]
    ulv = struct.unpack('<H', resp[14:16])[0]
    sex = resp[16:17]
    aid_int = struct.unpack('<I', aid)[0]
    print(f"  ✅ Logged in! AID={aid_int}")
    
    # Step 2: Connect to char server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect((SERVER, CHAR_PORT))
    except Exception as e:
        print(f"  ❌ Char server: {e}")
        return
    print("  ✅ Char server connected")
    
    # Send game_login (0x0065): a4(aid), a4(sid), a4(sid2), v(ulv), C(sex)
    game_pkt = struct.pack('<H', 0x0065)
    game_pkt += aid + sid + sid2
    game_pkt += struct.pack('<H', ulv)
    game_pkt += sex
    sock.sendall(game_pkt)
    print("  📤 Sent game_login (0x0065)")
    
    time.sleep(2)
    char_resp = recv_all(sock, 5)
    print(f"  📦 Char response: {len(char_resp)} bytes")
    
    # Parse character blocks (charBlockSize=155)
    chars = []
    pos = 0
    while pos + 2 <= len(char_resp):
        pk = struct.unpack('<H', char_resp[pos:pos+2])[0]
        
        # 0x099D - character info per page
        if pk == 0x099D and pos + 4 <= len(char_resp):
            data_start = pos + 4
            for c in range(3):  # 3 chars per page
                bs = data_start + c * 155
                if bs + 155 > len(char_resp):
                    break
                block = char_resp[bs:bs+155]
                # unpack with format matching charBlockSize=155
                # a4 V2 V V2 V6 v V2 v4 V v9 Z24 C8 v Z16 V4 C
                cid = block[0:4]
                name = block[66:90].split(b'\x00')[0].decode('ascii', errors='replace')
                slot = struct.unpack('<H', block[112:114])[0]
                lv = struct.unpack('<H', block[60:62])[0]
                job = struct.unpack('<H', block[58:60])[0]
                exp = struct.unpack('<I', block[4:8])[0]
                
                chars.append({'cid': cid, 'name': name, 'slot': slot, 'lv': lv, 'job': job, 'exp': exp})
                if name:
                    print(f"  👤 Slot {slot}: '{name}' (lv={lv}, job={job})")
                elif lv > 0 or exp > 0:
                    print(f"  👤 Slot {slot}: (lv={lv}, job={job})")
                else:
                    print(f"  🗑️  Slot {slot}: BROKEN (empty name)")
        
        # 0x006B - character list (all characters in one packet)
        if pk == 0x006B and pos + 2 <= len(char_resp):
            data_start = pos + 4  # skip packet ID and some header
            block_count = (len(char_resp) - data_start) // 155
            print(f"  📋 Char list: {block_count} blocks")
            
            for c in range(block_count):
                bs = data_start + c * 155
                if bs + 155 > len(char_resp):
                    break
                block = char_resp[bs:bs+155]
                cid = block[0:4]
                name = block[66:90].split(b'\x00')[0].decode('ascii', errors='replace')
                slot = struct.unpack('<H', block[112:114])[0]
                lv = struct.unpack('<H', block[60:62])[0]
                job = struct.unpack('<H', block[58:60])[0]
                exp = struct.unpack('<I', block[4:8])[0]
                
                if name:
                    print(f"  👤 Slot {slot}: '{name}' (lv={lv}, job={job})")
                    chars.append({'cid': cid, 'name': name, 'slot': slot, 'lv': lv, 'job': job, 'exp': exp})
                elif exp > 0 or lv > 0:
                    print(f"  👤 Slot {slot}: (lv={lv}, job={job})")
                else:
                    print(f"  🗑️  Slot {slot}: EMPTY-NAMED BROKEN CHARACTER")
                    chars.append({'cid': cid, 'name': '', 'slot': slot, 'lv': 0, 'job': 0, 'exp': 0, 'broken': True})
        
        pos += 2
        if pos > 200:
            break
    
    # Delete broken characters
    for ch in chars:
        if ch.get('broken'):
            cid = ch['cid']
            cid_hex = cid.hex()
            print(f"  🗑️  Deleting char slot {ch['slot']} (ID: {cid_hex})...")
            
            # char_delete2 (0x0827): send charID only
            del_pkt = struct.pack('<H', 0x0827) + cid
            sock.sendall(del_pkt)
            print("  📤 Sent char_delete2 (0x0827)")
            
            time.sleep(1.5)
            del_resp = recv_all(sock, 3)
            if del_resp:
                dpk = struct.unpack('<H', del_resp[0:2])[0]
                print(f"  📦 Delete response: 0x{dpk:04X} ({len(del_resp)} bytes)")
                
                if dpk == 0x0828:
                    result = del_resp[2]
                    if result == 1:
                        print("  ✅ Character deleted successfully!")
                    elif result == 0:
                        print("  ⚠️  Already planned for deletion")
                    else:
                        print(f"  ⚠️  Delete result: {result}")
                        # Try char_delete2_accept (0x0829) with empty code
                        accept_pkt = struct.pack('<H', 0x0829) + cid + b'\x00' * 6
                        sock.sendall(accept_pkt)
                        print("  📤 Sent char_delete2_accept (0x0829)")
                        time.sleep(1)
                        accept_resp = recv_all(sock, 2)
                        if accept_resp:
                            print(f"  📦 Accept response: 0x{struct.unpack('<H', accept_resp[0:2])[0]:04X}")
    
    # Check if we need to create a character
    valid_chars = [c for c in chars if c.get('name', '') and not c.get('broken')]
    
    if not valid_chars:
        print(f"  ✨ Creating character '{username}'...")
        
        # char_create (0x0067): a24(name), C7(str,agi,vit,int,dex,luk,slot), v2(hair_color,hair_style)
        create_pkt = struct.pack('<H', 0x0067)
        create_pkt += username.encode('ascii').ljust(24, b'\x00')[:24]  # name
        create_pkt += struct.pack('B', 1)   # str
        create_pkt += struct.pack('B', 9)   # agi
        create_pkt += struct.pack('B', 1)   # vit
        create_pkt += struct.pack('B', 1)   # int
        create_pkt += struct.pack('B', 9)   # dex
        create_pkt += struct.pack('B', 1)   # luk
        create_pkt += struct.pack('B', 0)   # slot
        create_pkt += struct.pack('<H', 0)  # hair_color
        create_pkt += struct.pack('<H', 0)  # hair_style
        
        sock.sendall(create_pkt)
        print("  📤 Sent char_create (0x0067)")
        
        time.sleep(1.5)
        create_resp = recv_all(sock, 5)
        if create_resp:
            ck = struct.unpack('<H', create_resp[0:2])[0]
            print(f"  📦 Create response: 0x{ck:04X} ({len(create_resp)} bytes)")
            
            if ck == 0x006D:
                result = create_resp[2]
                if result == 0:
                    print("  ✅ Character created successfully!")
                else:
                    print(f"  ⚠️  Create result: {result}")
            elif ck == 0x006B:
                print("  ✅ Character list received — character exists!")
            else:
                print("  ⚠️  Check response...")
    else:
        print(f"  ✅ Account has {len(valid_chars)} valid character(s)")
    
    sock.close()
    print("  ✅ Done")


# Main
for i in range(4, 12):
    fix_account(f'kicapmasin{i}')
    time.sleep(2)

print(f"\n{'='*50}")
print("  All accounts processed.")
print(f"  Then restart: cd /home/lot399/openkore-ai-v3 && ./start.sh")
print(f"{'='*50}")
