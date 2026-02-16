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
clippy-fix:
    @echo "Running clippy with automatic fixes..."
    @cargo clippy --fix --allow-dirty --allow-staged
    @echo "Clippy fixes applied."

# Run tests
test:
    @echo "Running tests..."
    @cargo test
    @echo "Tests complete."

# Generate test coverage report
coverage:
    @echo "Generating test coverage report..."
    @cargo llvm-cov --html
    @echo "Coverage report generated. Open target/llvm-cov/html/index.html to view."

# Count lines of code and documentation
count-lines:
    @echo "Counting lines of code and documentation..."
    @tokei --hidden --exclude target
    @echo "Line count complete."
