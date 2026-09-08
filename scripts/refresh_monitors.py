#!/usr/bin/env python3

import subprocess
import time
import sys

def main():
    print('=========================================')
    print('    MONITOR RE-PROBE & RE-ALIGN SCRIPT   ')
    print('=========================================')
    try:
        print('Applying screen alignment via xrandr...')
        xrandr_cmd = 'xrandr --output eDP-1 --auto --primary --output DP-1 --auto --right-of eDP-1 --output HDMI-1-0 --auto --right-of DP-1'
        subprocess.run(xrandr_cmd, shell=True, check=True)
        print('\nDisplay configuration successfully restored!')
        print('Layout: eDP-1 (Primary) -> DP-1 -> HDMI-1-0')
        print('=========================================')
    except subprocess.CalledProcessError as e:
        print(f'\n[Error] Failed to execute xrandr command: {e}', file=sys.stderr)
    except Exception as e:
        print(f'\n[Error] Unexpected error: {e}', file=sys.stderr)
if __name__ == '__main__':
    main()
