#!/usr/bin/env python3
"""
Test runner script for the Ollama Chat TUI application.
Run this script to execute all tests and generate a comprehensive report.
"""

import os
import sys
import subprocess
import importlib.util


def install_dependencies():
    """Install required test dependencies."""
    dependencies = [
        'pytest',
        'pytest-asyncio',
        'httpx',
        'textual'
    ]

    print("📦 Installing test dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"  ✅ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {dep}")
    print()


def run_tests():
    """Run the comprehensive test suite."""
    print("🔧 OLLAMA CHAT TUI - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()

    # Check if we can import required modules
    try:
        import pytest
        import asyncio
        print("✅ All required test modules available")
    except ImportError as e:
        print(f"❌ Missing required module: {e}")
        print("📦 Installing dependencies...")
        install_dependencies()

    # Import and run the test suite
    try:
        # Load the test module
        spec = importlib.util.spec_from_file_location("test_app_comprehensive", "test_app_comprehensive.py")
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)

        # Run all tests
        report = test_module.run_all_tests()

        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        print(report)

        # Save report
        with open("test_report.txt", "w") as f:
            f.write(report)

        print(f"\n📄 Report saved to: test_report.txt")
        print("\n🔧 Next steps:")
        print("1. Review the test report above")
        print("2. Send the test_report.txt file for analysis")
        print("3. Address any failing tests or errors")
        print("4. Run tests again after fixes")

        return True

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--install-deps":
            install_dependencies()
            return
        elif sys.argv[1] == "--debug":
            print("🔍 Running debug mode...")
            try:
                import test_debug_runner
                test_debug_runner.main()
            except ImportError:
                print("❌ Debug runner not found. Make sure test_debug_runner.py is in the same directory.")
            return

    success = run_tests()
    if not success:
        print("\n💡 TIP: If you're having import issues, try running:")
        print("   python run_tests.py --debug")
        print("   This will help identify what's missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()