# DeepChArUco

DeepChArUco is tracked as a pinned Git submodule.

The verified source tree is located at:

```text
third_party/deepcharuco/upstream
```

NanoDeepChArUco loads the inference modules from:

```text
third_party/deepcharuco/upstream/src
```

Pinned revision:

```text
37d569fc582b790843dce408c14556747927711c
```

Initialize the dependency with:

```bash
git submodule update --init --recursive
```

The repository `setup.sh` script performs this initialization and verifies
the pinned revision automatically.
