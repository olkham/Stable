#!/usr/bin/env bash
# Stable installer (macOS / Linux, bash or zsh).
#
# SOURCE this script so the virtual environment is still active afterwards:
#
#     source ./install.sh
#
# Running it as ./install.sh installs everything correctly too, but a child
# process cannot activate a venv in its parent shell, so it will just tell you
# the activate command at the end.
#
# Deliberately no `set -e`: when sourced, a failing command under `set -e`
# would close the user's shell. Every step is checked explicitly instead.

# ------------------------------------------------------- sourced or executed?
_stable_sourced=0
if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in *:file*) _stable_sourced=1 ;; esac
    _stable_here=${${(%):-%x}:A:h}
elif [ -n "${BASH_VERSION:-}" ]; then
    (return 0 2>/dev/null) && _stable_sourced=1
    _stable_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
else
    _stable_here=$(cd "$(dirname "$0")" && pwd)
fi

_stable_install() {
    local here="$1"
    local cand=() names=() vers=() paths=()
    local name path ver i choice pyexe venvpy

    printf '\n=== Stable installer ===\n\n'

    # ------------------------------------------------------------ find python
    for name in python python3; do
        command -v "$name" >/dev/null 2>&1 || continue
        # Resolve the real interpreter: `python` and `python3` are usually the
        # same binary behind two names, and asking the user to choose between
        # two identical options is just a worse way of saying nothing.
        path=$("$name" -c 'import sys;print(sys.executable)' 2>/dev/null) || continue
        [ -n "$path" ] || continue
        for i in "${paths[@]}"; do
            [ "$i" = "$path" ] && path="" && break
        done
        [ -n "$path" ] || continue
        ver=$("$name" -c 'import platform;print(platform.python_version())' 2>/dev/null)
        names+=("$name"); paths+=("$path"); vers+=("$ver")
    done

    if [ "${#names[@]}" -eq 0 ]; then
        printf 'ERROR: neither "python" nor "python3" was found on your PATH.\n'
        printf 'Install Python 3.11 or newer, then run this again.\n'
        return 1
    fi

    if [ "${#names[@]}" -eq 1 ]; then
        pyexe="${names[0]}"
        printf 'Using the only Python found: %s (%s)\n  %s\n' \
            "${names[0]}" "${vers[0]}" "${paths[0]}"
    elif [ ! -t 0 ]; then
        # Non-interactive (piped/CI): take the first rather than hang on a prompt.
        pyexe="${names[0]}"
        printf 'Not a terminal, so defaulting to %s (%s).\n' "${names[0]}" "${vers[0]}"
    else
        printf 'Two Python interpreters are available:\n\n'
        for i in "${!names[@]}"; do
            printf '  [%d] %-8s %s\n      %s\n' \
                "$((i + 1))" "${names[$i]}" "${vers[$i]}" "${paths[$i]}"
        done
        printf '\n'
        printf 'Create the virtual environment with which one? [1/2] (default 1): '
        read -r choice
        case "$choice" in
            2) pyexe="${names[1]}" ;;
            *) pyexe="${names[0]}" ;;
        esac
        printf '\nUsing %s.\n' "$pyexe"
    fi

    # Stable needs 3.11+ (pyproject: requires-python = ">=3.11").
    if ! "$pyexe" -c 'import sys;sys.exit(0 if sys.version_info>=(3,11) else 1)'; then
        printf '\nERROR: %s is too old (%s). Stable needs Python 3.11 or newer.\n' \
            "$pyexe" "$("$pyexe" -c 'import platform;print(platform.python_version())')"
        return 1
    fi

    # ------------------------------------------------------- create the venv
    # Layout differs by platform: bin/ on macOS and Linux, Scripts/ on Windows
    # (this script still runs there under Git Bash / MSYS).
    if [ -f "$here/.venv/bin/activate" ] || [ -f "$here/.venv/Scripts/activate" ]; then
        printf '\nReusing the existing .venv (delete it yourself if you want a clean one).\n'
    else
        printf '\nCreating .venv ...\n'
        if ! "$pyexe" -m venv "$here/.venv"; then
            printf '\nERROR: could not create the virtual environment.\n'
            printf 'On Debian/Ubuntu the venv module ships separately: apt install python3-venv\n'
            return 1
        fi
    fi

    if [ -x "$here/.venv/bin/python" ]; then
        venvpy="$here/.venv/bin/python"
        _stable_activate="$here/.venv/bin/activate"
        _stable_activate_hint='source .venv/bin/activate'
    elif [ -x "$here/.venv/Scripts/python.exe" ]; then
        venvpy="$here/.venv/Scripts/python.exe"
        _stable_activate="$here/.venv/Scripts/activate"
        _stable_activate_hint='source .venv/Scripts/activate'
    else
        printf '\nERROR: no interpreter under %s/.venv - it looks broken.\n' "$here"
        printf 'Delete the .venv directory and run this script again.\n'
        return 1
    fi

    # ---------------------------------------------------------- pip + install
    printf '\nUpgrading pip if needed ...\n'
    if ! "$venvpy" -m pip install --upgrade pip; then
        printf '\nERROR: upgrading pip failed. Check your network or proxy settings.\n'
        return 1
    fi

    printf '\nInstalling Stable and its dependencies ...\n'
    # Install from inside the project dir: an absolute "path[extras]" argument
    # is not portable (under Git Bash the MSYS path reaches Windows pip in a
    # form it rejects), whereas ".[all]" is understood everywhere.
    if ! ( cd "$here" && "$venvpy" -m pip install -e ".[all]" ); then
        printf '\nERROR: installing Stable failed - see the pip output above.\n'
        return 1
    fi

    return 0
}

_stable_install "$_stable_here"
_stable_rc=$?

if [ "$_stable_rc" -eq 0 ]; then
    if [ "$_stable_sourced" -eq 1 ]; then
        # shellcheck disable=SC1091
        . "$_stable_activate"
        _stable_state='installed and this environment is now ACTIVE.'
    else
        _stable_state="installed, but this shell is NOT in the venv (you ran the
 script instead of sourcing it). Activate it with:

     $_stable_activate_hint"
    fi

    cat <<EOF

============================================================
 Stable is $_stable_state

 Start the app with:

     stable serve

 That serves the web UI and opens your browser. Other things
 worth knowing:

     stable sample      run the bundled demo clip end to end
     stable trackers    list the tracking algorithms available
     stable --help      every command

 GPU tracking (CoTracker3 / TAPIR) needs PyTorch, which is
 installed separately - see docs/windows-gpu-setup.md for the
 CUDA wheel selection (it applies to Linux too).

 Next time, reactivate with:  $_stable_activate_hint
============================================================

EOF
fi

# Sourced scripts share the user's shell, so don't leave our scratch names behind.
unset -f _stable_install 2>/dev/null
unset _stable_here _stable_sourced _stable_state _stable_rc 2>/dev/null
unset _stable_activate _stable_activate_hint 2>/dev/null
