#!/usr/bin/env python
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import create_app


def main():
    parser = argparse.ArgumentParser(description='MAV - Motion Amplification Visualization')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    config_name = 'production' if args.production else 'development'
    app = create_app(config_name)
    debug = args.debug or (not args.production)
    
    app.run(host=args.host, port=args.port, debug=debug)


if __name__ == '__main__':
    main()
