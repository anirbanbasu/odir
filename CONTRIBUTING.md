# Contributing

Thank you for your interest in contributing to the Ollama Downloader in Rust (ODIR). Pull requests are welcome. However, please open an issue first to discuss what you would like to change.

We will, in the future, update this document with more detailed contribution guidelines.

## Just recipes for development

To set up a development environment for ODIR, you can follow these steps.

1. **Clone the Repository:**
    If you haven't already, clone the ODIR repository to your local machine:

   ```bash
   git clone https://github.com/anirbanbasu/odir.git
   ```

2. **Install Rust toolchain:**
   If you haven't already, install Rust using [the instructions](https://rust-lang.org/tools/install/).

3. **Install necessary tools:**
    - _just_ is a handy command runner, which you can install using [the instructions](https://github.com/casey/just?tab=readme-ov-file#installation).
    - _prek_ is a fast Rust-based alternative to `pre-commit`, which you can install using [the instructions](https://prek.j178.dev/installation/).
    - _osv-scanner_ is a tool for scanning dependencies for vulnerabilities, which you can install using [the instructions](https://google.github.io/osv-scanner/installation/).

4. **Set up pre-commit hooks:**
   To ensure code quality and consistency, we use `pre-commit` hooks. You can set them up by running:

   ```bash
   just install-pre-commit-hooks
   ```

5. **Install tools for development:**
   Run the `just` recipe to install all necessary tools for development.

   ```bash
   just install-tools
   ```

6. **Check available `just` recipes:**
   You can check the available `just` recipes for various development tasks by running:

   ```bash
   just -l
   ```


## Licensing & Contributions

By contributing to this project, you agree to the following:

1. **License:** Your contributions will be licensed under the **MIT License**.
2. **Media/Docs:** You grant the maintainers the right to re-license your documentation and media assets, but not the source code, under a more permissive licenses in the future.
3. **Developer Certificate of Origin (DCO):** To ensure a clear chain of ownership, we strongly suggest all commits to be "signed-off."

### Developer Certificate of Origin (DCO)

By adding `Signed-off-by: Your Name <email@example.com>` to your commit message, you certify that you have the right to submit the work under the terms of the [Developer Certificate of Origin 1.1](https://developercertificate.org).

To protect your privacy, you may use your _GitHub-provided no-reply email address_ or any other aliases for your signatures.

If you use the GitHub-provided no-reply email, every signed-off commit, will look similar to `Signed-off-by: Real Name <username@users.noreply.github.com>`.
