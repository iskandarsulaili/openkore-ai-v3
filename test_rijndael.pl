#!/usr/bin/env perl
# Test Crypt::Rijndael for password encryption
use strict;
use Crypt::Rijndael;

my $key24 = pack("C24", (6, 169, 33, 64, 54, 184, 161, 91, 81, 46, 3, 213, 52, 18, 0, 6, 61, 175, 186, 66, 157, 158, 180, 48));
my $key32 = pack("C32", (0x06, 0xA9, 0x21, 0x40, 0x36, 0xB8, 0xA1, 0x5B, 0x51, 0x2E, 0x03, 0xD5, 0x34, 0x12, 0x00, 0x06, 0x06, 0xA9, 0x21, 0x40, 0x36, 0xB8, 0xA1, 0x5B, 0x51, 0x2E, 0x03, 0xD5, 0x34, 0x12, 0x00, 0x06));
my $data = pack("a24", "b0tTib0tTi");

print "Crypt::Rijndael version: $Crypt::Rijndael::VERSION\n";

eval {
    my $cipher = Crypt::Rijndael->new($key24, Crypt::Rijndael::MODE_ECB());
    print "24-byte key accepted\n";
    print "Block size: " . $cipher->blocksize() . " bytes\n";
    # Try to encrypt 24-byte data
    if ($cipher->blocksize() == 24) {
        my $enc = $cipher->encrypt($data);
        print "Encrypted 24B: " . unpack("H*", $enc) . "\n";
    } else {
        # Standard 16-byte block - need to handle differently
        print "Standard AES (16B block), not Rijndael-192\n";
        my $enc1 = $cipher->encrypt(substr($data, 0, 16));
        print "Encrypted 16B: " . unpack("H*", $enc1) . "\n";
    }
};
if ($@) { print "Error (24-byte key): $@\n"; }

eval {
    my $cipher = Crypt::Rijndael->new($key32, Crypt::Rijndael::MODE_ECB());
    print "\n32-byte key accepted\n";
    print "Block size: " . $cipher->blocksize() . " bytes\n";
};
if ($@) { print "Error (32-byte key): $@\n"; }
