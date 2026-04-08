# Build the project in debug mode
build-debug:
    @echo "Building project in debug mode..."
    @cargo build
    @echo "Debug build complete."

# Build the project in release mode
build-release:
    @echo "Building project in release mode..."
    @cargo build --release
    @echo "Release build complete."

# Install required cargo tools
install-tools:
    @echo "Installing required cargo and Rust tools..."
    @cargo install tokei
    @cargo install cargo-edit
    @cargo install cargo-llvm-cov
    @rustup component add clippy
    @echo "Cargo and Rust tools installed."

# Install pre-commit hooks using 'prek'
install-pre-commit-hooks:
    @echo "Installing pre-commit hooks using prek..."
    @prek install
    @echo "Pre-commit hooks installed."

# Update pre-commit hooks using 'prek'
pre-commit-update:
    @echo "Updating pre-commit hooks using prek..."
    @prek auto-update
    @echo "Pre-commit hooks updated."

# Upgrade project dependencies using 'cargo'
upgrade-dependencies:
    @echo "Upgrading project dependencies..."
    @cargo update --verbose
    @echo "Dependencies upgraded."

# Bump the patch version of the project using 'cargo'
bump-patch:
    @echo "Current project version: $(cargo pkgid | cut -d# -f2)"
    @cargo set-version --bump patch
    @echo "Updated project to: $(cargo pkgid | cut -d# -f2)"

# Format the code
format:
    @echo "Formatting code..."
    @cargo fmt
    @echo "Code formatted."

# Run the type checker and linter
type-check-and-lint:
    @echo "Running type checker and linter..."
    @cargo check
    @cargo clippy -- -D warnings
    @echo "Type checking and linting complete."

# Fix with clippy
type-check-and-lint-fix:
    @echo "Running clippy with automatic fixes..."
    @cargo clippy --fix --allow-dirty --allow-staged
    @echo "Clippy fixes applied."

# Generate crate documentation
generate-docs-and-show-in-browser:
    @echo "Cleaning old documentation..."
    @rm -rf target/doc
    @echo "Generating crate documentation..."
    @cargo doc --no-deps --open
    @echo "Documentation generated and opened in browser."

export RUN_INTEGRATION_TESTS := "1"

# Run comprehensive tests, including downloading models
test-comprehensive $RUN_INTEGRATION_TESTS="1":
    @echo "Running all tests, including downloading models, with --test-threads=1 to avoid shared-state conflicts in integration downloads..."
    @cargo test -- --include-ignored --test-threads=1
    @echo "All tests complete."

# Generate comprehensive test coverage report (including integration tests) and show in browser
coverage-comprehensive-and-show-in-browser $RUN_INTEGRATION_TESTS="1":
    @echo "Generating comprehensive test coverage report (including integration tests) with --test-threads=1 for stable integration coverage, and opening it in browser..."
    @cargo llvm-cov --all-targets --html --open -- --include-ignored --test-threads=1
    @echo "Comprehensive coverage report generated and opened in browser."

# Run tests
test $RUN_INTEGRATION_TESTS="0":
    @echo "Running tests..."
    @cargo test
    @echo "Tests complete."

# Generate test coverage report and show in browser
coverage-and-show-in-browser $RUN_INTEGRATION_TESTS="0":
    @echo "Generating test coverage report and opening it in browser..."
    @cargo llvm-cov --all-targets --html --open
    @echo "Coverage report generated and opened in browser."

# Count lines of code and documentation
count-lines:
    @echo "Counting lines of code and documentation..."
    @tokei --hidden --exclude target
    @echo "Line count complete."

# Run the Open Source Vulnerability scanner
vulnerability-scan:
    @echo "Running Open Source Vulnerability scanner..."
    @osv-scanner scan source -r .
    @echo "Vulnerability scan complete."

# Publish the crate to crates.io
publish:
    @echo "Publishing crate to crates.io..."
    @cargo publish --locked
    @echo "Crate published to crates.io."

# Advisory helper for Homebrew formula updates.
# It does NOT edit any formula. It prints the lines to update.
#
# Required:
#   release_tag: tag used in the source archive URL (for example: v0.1.1).
# Optional:
#   repo_base_url: repository base URL (for example: https://github.com/anirbanbasu/odir).
#                  If omitted, this recipe derives it from `git remote get-url origin`.
#   formula_url: publicly accessible raw formula URL for showing current url/sha256 values.
#
# Example:
# just brew-formula-advise v0.1.1
# or
# just brew-formula-advise v0.1.1 https://github.com/anirbanbasu/odir https://raw.githubusercontent.com/anirbanbasu/homebrew-tap/refs/heads/main/Formula/odir.rb

# Advisory helper for Homebrew formula updates.
brew-formula-advise release_tag repo_base_url="" formula_url="":
    #!/usr/bin/env bash
    set -euo pipefail

    release_tag="{{ release_tag }}"
    repo_base_url="{{ repo_base_url }}"
    formula_url="{{ formula_url }}"

    if [[ -z "$repo_base_url" ]]; then
        origin_url="$(git remote get-url origin)"
        if [[ "$origin_url" =~ ^https?:// ]]; then
            repo_base_url="${origin_url%.git}"
        elif [[ "$origin_url" =~ ^git@([^:]+):(.+)\.git$ ]]; then
            repo_base_url="https://${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        elif [[ "$origin_url" =~ ^ssh://git@([^/]+)/(.+)\.git$ ]]; then
            repo_base_url="https://${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        else
            echo "Could not derive repository base URL from origin: $origin_url" >&2
            echo "Pass repo_base_url explicitly." >&2
            exit 1
        fi
    fi

    if [[ "$repo_base_url" =~ ^https://github\.com/([^/]+)/([^/]+)$ ]]; then
        gh_owner="${BASH_REMATCH[1]}"
        gh_repo="${BASH_REMATCH[2]}"
        gh_release_api_url="https://api.github.com/repos/$gh_owner/$gh_repo/releases/tags/$release_tag"
        if ! curl -fsSL "$gh_release_api_url" >/dev/null 2>&1; then
            echo "Failed to validate GitHub release tag '$release_tag' via $gh_release_api_url." >&2
            echo "The tag may be invalid, or the lookup may have failed due to rate limiting or network/API issues." >&2
            exit 1
        fi
    fi

    release_tarball_url="$repo_base_url/archive/refs/tags/$release_tag.tar.gz"

    hash_cmd=""
    if command -v shasum >/dev/null 2>&1; then
        hash_cmd='shasum -a 256'
    elif command -v sha256sum >/dev/null 2>&1; then
        hash_cmd='sha256sum'
    else
        echo "Neither shasum nor sha256sum is available." >&2
        exit 1
    fi

    echo "Repository base URL: $repo_base_url"
    echo "Computed release archive URL: $release_tarball_url"

    echo "Computing sha256 for release archive: $release_tarball_url"
    tmp_archive="$(mktemp)"
    trap 'rm -f "$tmp_archive"' EXIT

    if ! curl -fsSL -o "$tmp_archive" "$release_tarball_url" >/dev/null 2>&1; then
        echo "Failed to download release tarball: $release_tarball_url" >&2
        exit 1
    fi

    if [[ "$hash_cmd" == "shasum -a 256" ]]; then
        new_sha256="$(shasum -a 256 "$tmp_archive" | awk '{print $1}')"
    else
        new_sha256="$(sha256sum "$tmp_archive" | awk '{print $1}')"
    fi

    current_url=""
    current_sha=""
    current_sha_value=""

    if [[ -n "$formula_url" ]]; then
        formula_body="$(curl -fsSL "$formula_url")"
        current_url="$(printf '%s\n' "$formula_body" | awk '/^[[:space:]]*url[[:space:]]+"/{print; exit}')"
        current_sha="$(printf '%s\n' "$formula_body" | awk '/^[[:space:]]*sha256[[:space:]]+"/{print; exit}')"
        current_sha_value="$(printf '%s\n' "$current_sha" | cut -d'"' -f2 | tr 'A-F' 'a-f')"

        if [[ -n "$current_sha_value" && "$current_sha_value" == "$new_sha256" ]]; then
            echo "Your Homebrew formula does not require any changes."
            exit 0
        fi
    fi

    echo
    echo "Write these values into your formula:"
    echo "  url \"$release_tarball_url\""
    echo "  sha256 \"$new_sha256\""
    echo

    if [[ -n "$formula_url" ]]; then
        echo "Inspecting current values from formula URL: $formula_url"

        if [[ -n "$current_url" || -n "$current_sha" ]]; then
            echo
            echo "Current formula values (first match):"
            [[ -n "$current_url" ]] && echo "  $current_url"
            [[ -n "$current_sha" ]] && echo "  $current_sha"
        else
            echo "Could not find url/sha256 lines in formula content."
        fi
    fi
