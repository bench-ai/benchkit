[![](https://bench-ai.com/static/media/bench.f0b00cb77f69869f37586406c4ff9ebe.svg)](https://bench-ai.com/)

# Contributing to `benchkit`

---

We at `Bench AI` love open source and welcome everyone to contribute to our repository!
Below we will detail how you can go about doing this!

## What we need help with

1. BUGS 🐛

   1. We could always use the help in squashing any bugs

2. Integrations 🔨

   1. We would love for integrations to be added to your favorite libraries / products: Ray, Snowflake, Tensorflow, etc...

3. Optimizations 🚀

   1. Think Something is too slow go ahead and speed it up!

4. Graphing 📊

   1. We are missing out on a bunch of cool graphs that should be directly integrated, please add them

5. Misc 🤷‍
   1. Have a cool feature that doesn't fit into any of these categories? Go ahead and do it we would love to see what you build!

## Contribution Steps

1. [Build a Local Copy](#build-from-source)
2. Create a new branch
3. [Add your contributions](#contributing-your-code)
4. Create a Pull Request in the [benchkit](https://github.com/bench-ai/benchkit/pulls) repo
5. Celebrate once your code gets merged!

### Build From Source

- Start off by forking your own personal copy of `benchkit`
  ```shell
  git clone https://github.com/<your-username>/benchkit.git
  git remote add upstream https://github.com/bench-ai/benchkit.git
  ```
- Now build the local version

  - Requirements
    - python >= 3.10
    - OS: MacOS, Linux, Windows
  - Make a virtual environment
    ```shell
    python3.10 -m venv venv
    source venv/bin/activate
    ```
  - Install poetry(the build tool)
    ```shell
    pip install poetry
    ```
  - Install all requirements
    ```shell
    poetry install
    ```

### Contributing your code

- Only code in the `src` folder
- Write appropriate tests in the `tests` folder
- Before you commit your code it must meet some requirements

  1. It must pass all tests
  2. It must be formatted following `black` standards

  3. Run this command to ensure all tests are passed, this will
     also ensure your code gets converted to the `black` format
     ```shell
     nox
     ```
  4. flake8 standards are also followed so any issues documented during the nox
     process must be resolved.
