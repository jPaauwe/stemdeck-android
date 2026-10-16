# Signing

Android only installs an update over an existing app when both are signed
with the same key. CI reads the key from the repository secret
`SIGNING_KEYSTORE_BASE64` (a base64-encoded PKCS12 keystore). The keystore
itself is the secret; its password, alias and key password default to
`stemdeck-sideload` / `sideload` / `stemdeck-sideload` unless the secrets
`SIGNING_KEYSTORE_PASSWORD`, `SIGNING_KEY_ALIAS` and `SIGNING_KEY_PASSWORD`
are set.

Without the secret, builds are signed with a throwaway debug key, and each
one has to be installed after uninstalling the previous one.

Never commit a keystore to this repository.
