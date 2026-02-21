# List available commands by default
_default:
    @just --list

# List available test scripts
list:
    @for f in tests/*.js; do printf "  %s\n" "$(basename "$f" .js)"; done

# Start the server with bunyan pretty-print
test script:
    @node --max-old-space-size=16384 tests/{{ script }}.js

# Start web server for html tests
web:
    http-server .
