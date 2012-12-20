#include <mm_plugin.h>
#include <yubikey.h>
#include <ykdef.h>
#include <ykpers.h>
#include <stdlib.h>
#include <time.h>

/****
YubiKey Challenge-Response Authentication

To configure YubiKeys for mm usage:
1) Download the cross platform personalization tool from http://yubico.com/personalization-tool.
2) Insert YubiKey and launch the personalization tool (may require sudo access).
3) Click "Challenge-Response Mode".
4) Click "Yubico OTP".
5) Select "Configuration Slot 2"
6) In the Private Identity text box enter "21 92 78 11 55 8a".
7) In the Secret Key text box enter "e7 32 df 49 f3 87 e6 89 04 d2 03 6a 59 ad b7 2f".
8) Click "Write Configuration".
9) Done!

Unix implementation derived from "ykchalresp.c" in ykpers repository.
Windows implementation derived from "MFCTestDlg.cpp" in Yubikey Client API installer.

!!! Attention Linux Users !!!
cp trunk/3rdparty/ykpers-1.6.3/70-yubikey.rules /etc/udev/rules.d

!!! Attention Windows Users !!!
Install Yubikey Client API.
****/

using namespace mm;

static int challenge_response(YK_KEY *yk, int slot,
                              unsigned char *challenge, unsigned int len,
                              bool hmac, bool may_block, bool verbose, unsigned char output_buf[(SHA1_MAX_BLOCK_SIZE * 2) + 1])
{
    unsigned char response[64];
    int yk_cmd;
    unsigned int flags = 0;
    unsigned int response_len = 0;
    unsigned int expect_bytes = 0;

    memset(response, 0, sizeof(response));

    if (may_block)
        flags |= YK_FLAG_MAYBLOCK;

    if (verbose) {
        fprintf(stderr, "Sending %i bytes %s challenge to slot %i\n", len, (hmac == true) ? "HMAC" : "Yubico", slot);
        //_yk_hexdump(challenge, len);
    }

    switch(slot) {
    case 1:
        yk_cmd = (hmac == true) ? SLOT_CHAL_HMAC1 : SLOT_CHAL_OTP1;
        break;
    case 2:
        yk_cmd = (hmac == true) ? SLOT_CHAL_HMAC2 : SLOT_CHAL_OTP2;
        break;
    }

    if (!yk_write_to_key(yk, yk_cmd, challenge, len))
        return 0;

    if (verbose) {
        fprintf(stderr, "Reading response...\n");
    }

    /* HMAC responses are 160 bits, Yubico 128 */
    expect_bytes = (hmac == true) ? 20 : 16;

    if (! yk_read_response_from_key(yk, slot, flags,
                                    &response, sizeof(response),
                                    expect_bytes,
                                    &response_len))
        return 0;

    if (hmac && response_len > 20)
        response_len = 20;
    if (! hmac && response_len > 16)
        response_len = 16;

    memset(output_buf, 0, SHA1_MAX_BLOCK_SIZE * 2 + 1);
    if (hmac) {
        yubikey_hex_encode((char *)output_buf, (char *)response, response_len);
    } else {
        yubikey_modhex_encode((char *)output_buf, (char *)response, response_len);
    }
    // printf("%s\n", output_buf);

    return 1;
}

/*!
 * \ingroup initializers
 * \brief Initialize yubikey
 * \author Josh Klontz \cite jklontz
 */
class YubiKey : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        // Read from device
        YK_KEY *yk = 0;

        if (!yk_init())
            qFatal("YubiKey::initialize yk_init failure.");

        if (!(yk = yk_open_first_key()))
            qFatal("Could not connect to license.");

        // Challenge value is arbitrary
        srand(time(NULL));
        uint8_t challenge[6] = {rand()%255, rand()%255, rand()%255, rand()%255, rand()%255, rand()%255};
        unsigned char output_buf[(SHA1_MAX_BLOCK_SIZE * 2) + 1];
        if (!challenge_response(yk, 2, challenge, 6, false, true, false, output_buf))
            qFatal("YubiKey::initialize challenge_response failure.");

        if (yk && !yk_close_key(yk))
            qFatal("YubiKey::initialize yk_close_key failure.");

        if (!yk_release())
            qFatal("YubiKey::initialize yk_release failure.");

        // Check response
        // Our Secret Key! Shhh...
        const uint8_t key[YUBIKEY_KEY_SIZE] = {0xe7, 0x32, 0xdf, 0x49, 0xf3, 0x87, 0xe6, 0x89, 0x04, 0xd2, 0x03, 0x6a, 0x59, 0xad, 0xb7, 0x2f};
        yubikey_token_st out;
        yubikey_parse(output_buf, key, &out);

        // Our Private Identity! Shhh...
        uint8_t uid[YUBIKEY_UID_SIZE] = {0x21, 0x92, 0x78, 0x11, 0x55, 0x8a};
        if ((uid[0] != (out.uid[0] ^ challenge[0])) ||
            (uid[1] != (out.uid[1] ^ challenge[1])) ||
            (uid[2] != (out.uid[2] ^ challenge[2])) ||
            (uid[3] != (out.uid[3] ^ challenge[3])) ||
            (uid[4] != (out.uid[4] ^ challenge[4])) ||
            (uid[5] != (out.uid[5] ^ challenge[5])))
            qFatal("Invalid license.");
    }

    void finalize() const
    {
        // Nothing to do
    }
};

MM_REGISTER(Initializer,YubiKey,"")

#include "yubico.moc"
