# Scripts

Scripts are thin human-facing entrypoints that call package CLI modules under
`src/blockcipher_nd/cli/`.

Do not put training loops, dataset generation, feature extraction, cache logic,
or result validation implementations here.
