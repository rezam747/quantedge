# Define colors
RED="\[\033[0;31m\]"
GREEN="\[\033[0;32m\]"
YELLOW="\[\033[0;33m\]"
BLUE="\[\033[0;34m\]"
CYAN="\[\033[0;36m\]"
RESET="\[\033[0m\]"

# Function to show Git branch in prompt
parse_git_branch() {
    git branch 2>/dev/null | grep '*' | sed 's/* //'
}

# Function to show Python virtual environment
parse_virtual_env() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "($(basename $VIRTUAL_ENV))"
    fi
}

# Custom prompt with Git branch and Virtual Env in different colors
export PS1="$GREEN\u@\h$RESET:$BLUE\w$RESET $YELLOW\$(parse_virtual_env)$CYAN\$(parse_git_branch)$RESET\$ "

eval "$(/opt/homebrew/bin/brew shellenv)"

# === Auto-activate venv in ~/Documents/quantedge ===
function cd() {
    builtin cd "$@" || return
    
    # Check if we're in the quantedge directory
    if [[ "$PWD" == "/Users/mohammadrezamirzazadeh/Documents/quantedge"* ]]; then
        VENV_PATH="$PWD/.venv"
        
        # If virtual environment exists, activate it
        if [ -d "$VENV_PATH" ]; then
            source "$VENV_PATH/bin/activate"
        fi
    # Deactivate if we left the project directory
    elif [[ "$VIRTUAL_ENV" == *"quantedge"* ]]; then
        deactivate 2>/dev/null || true
    fi
}

# Initialize the function for the current shell
cd "$PWD"
