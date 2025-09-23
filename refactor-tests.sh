#!/bin/bash

# refactor-tests.sh
# Wrapper script to update tests after code refactoring using Gemini CLI
# Usage: ./refactor-tests.sh [additional context]

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v geminI >/dev/null 2>&1; then
        log_error "gemini-cli not found. Please install it first."
        exit 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        log_error "git not found. This script requires git."
        exit 1
    fi

    if ! command -v jq >/dev/null 2>&1; then
        log_error "jq not found. Please install jq for JSON processing."
        exit 1
    fi

    if [[ ! -f "$REPO_ROOT/scripts/analyze-refactor-context.sh" ]]; then
        log_error "Context analysis script not found at $REPO_ROOT/scripts/analyze-refactor-context.sh"
        exit 1
    fi

    log_success "All prerequisites met"
}

# Check if there are any changes to analyze
check_for_changes() {
    log_info "Checking for recent changes..."

    if ! git diff --quiet HEAD~1 HEAD 2>/dev/null; then
        log_success "Found changes in last commit"
    else
        log_warning "No changes found in last commit. Make sure you have committed your refactoring changes."
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
}

# Create output directories
prepare_output_directories() {
    log_info "Preparing output directories..."

    mkdir -p "$REPO_ROOT/md_current"

    log_success "Output directories ready"
}

# Run the refactor-tests command
run_refactor_command() {
    local additional_context="$1"

    log_info "Gathering refactor context..."

    # Combine additional context with the command
    local full_context="Refactor tests after code changes. Additional context: $additional_context"

    log_info "Executing Gemini CLI refactor-tests command..."

    # Check if refactor-tests.toml exists
    local toml_path="$REPO_ROOT/.gemini/commands/refactor-tests.toml"
    if [[ ! -f "$toml_path" ]]; then
        log_error "refactor-tests.toml not found at $toml_path"
        log_info "Please ensure the refactor_tests.toml configuration file is in your repo root"
        exit 1
    fi

    # Execute the gemini-cli command
    if gemini /refactor-tests "$full_context"; then
        log_success "Gemini CLI execution completed"
    else
        log_error "Gemini CLI execution failed"
        exit 1
    fi
}

# Verify outputs were created
verify_outputs() {
    log_info "Verifying outputs..."

    local success=true

    if [[ -f "$REPO_ROOT/md_current/test-refactor-summary.md" ]]; then
        log_success "Test refactor summary created"
    else
        log_warning "Test refactor summary not found"
        success=false
    fi

    if [[ -f "$REPO_ROOT/commit_message.txt" ]]; then
        log_success "Commit message created"

        # Show the commit message
        log_info "Generated commit message:"
        echo "----------------------------------------"
        cat "$REPO_ROOT/commit_message.txt"
        echo "----------------------------------------"
    else
        log_warning "Commit message not found"
        success=false
    fi

    if [[ "$success" == true ]]; then
        log_success "All expected outputs created successfully"
    else
        log_warning "Some outputs may be missing. Check the Gemini CLI execution logs."
    fi
}

# Run tests to verify everything is working
# run_final_test_check() {
#     log_info "Running final test check..."

#     # Try to run tests to see if they pass now
#     if command -v npm >/dev/null 2>&1 && [[ -f "$REPO_ROOT/package.json" ]]; then
#         if npm test >/dev/null 2>&1; then
#             log_success "All tests are now passing!"
#         else
#             log_warning "Some tests are still failing. Manual review may be needed."
#         fi
#     elif command -v yarn >/dev/null 2>&1 && [[ -f "$REPO_ROOT/package.json" ]]; then
#         if yarn test >/dev/null 2>&1; then
#             log_success "All tests are now passing!"
#         else
#             log_warning "Some tests are still failing. Manual review may be needed."
#         fi
#     else
#         log_info "Could not run tests automatically. Please run your test suite manually to verify."
#     fi
# }

# Show next steps
show_next_steps() {
    log_info "Next steps:"
    echo "1. Review the generated test changes"
    echo "2. Run your test suite: npm test (or equivalent)"
    echo "3. If tests pass, commit the changes:"
    echo "   git add ."
    echo "   git commit -F commit_message.txt"
    echo "4. If tests still fail, review the summary and make manual adjustments"
}

# Main execution
main() {
    local additional_context="${1:-No additional context provided}"

    log_info "Starting test refactoring workflow..."
    echo "Repository: $REPO_ROOT"
    echo "Additional Context: $additional_context"
    echo ""

    check_prerequisites
    check_for_changes
    prepare_output_directories
    run_refactor_command "$additional_context"
    verify_outputs
    # run_final_test_check
    show_next_steps

    log_success "Test refactoring workflow completed!"
}

# Handle script arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
