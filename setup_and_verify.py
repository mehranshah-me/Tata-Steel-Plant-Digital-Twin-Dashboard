"""
Setup and Verification Script
==============================

Verifies that all dependencies are installed and the system is working correctly.
Run this script after installation to ensure everything is set up properly.
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ❌ Python {version.major}.{version.minor} detected.")
        print("     Python 3.8 or higher required.")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    print("\nChecking dependencies...")
    
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'networkx': 'NetworkX',
        'sklearn': 'Scikit-learn',
        'statsmodels': 'Statsmodels',
        'plotly': 'Plotly',
        'streamlit': 'Streamlit'
    }
    
    all_installed = True
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name} not found")
            all_installed = False
            missing_packages.append(package)
    
    if not all_installed:
        print(f"\n  Missing packages: {', '.join(missing_packages)}")
        print("  Install with: pip install -r requirements.txt")
    
    return all_installed


def check_project_structure():
    """Verify project file structure."""
    print("\nChecking project structure...")
    
    required_files = [
        'safety_twin_simulation.py',
        'app.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file} not found")
            all_exist = False
    
    return all_exist


def verify_simulation():
    """Run a quick simulation test."""
    print("\nRunning quick simulation test...")
    
    try:
        from safety_twin_simulation import SteelPlantDigitalTwin
        
        # Run short simulation
        twin = SteelPlantDigitalTwin(seed=999)
        df = twin.run_simulation(days=7, workers_per_shift=50, shifts_per_day=3)
        
        if len(df) == 28:  # 7 days × 4 zones
            print("  ✓ Simulation test passed")
            print(f"    Generated {len(df)} records for 7 days")
            
            # Check KPIs
            kpis = twin.calculate_kpis(df)
            print(f"    LTIFR: {kpis['LTIFR']:.2f}")
            print(f"    Incidents: {int(kpis['Incident_Count'])}")
            
            return True
        else:
            print(f"  ❌ Unexpected record count: {len(df)}")
            return False
            
    except Exception as e:
        print(f"  ❌ Simulation test failed: {e}")
        return False


def check_data_directory():
    """Check if data directory and files exist."""
    print("\nChecking data directory...")
    
    data_dir = Path('data')
    if not data_dir.exists():
        print("  ℹ️  Data directory not found (will be created on first run)")
        return True
    
    print("  ✓ Data directory exists")
    
    data_files = [
        'simulated_safety_data.csv',
        'safety_kpis.json',
        'zone_kpis.csv'
    ]
    
    files_exist = False
    for file in data_files:
        path = data_dir / file
        if path.exists():
            size = path.stat().st_size
            print(f"    ✓ {file} ({size:,} bytes)")
            files_exist = True
        else:
            print(f"    ℹ️  {file} not found (will be generated)")
    
    return True


def run_full_verification():
    """Run complete system verification."""
    print("="*60)
    print("DIGITAL SAFETY TWIN - SYSTEM VERIFICATION")
    print("="*60)
    print()
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Project Structure': check_project_structure(),
        'Data Directory': check_data_directory(),
        'Simulation Test': False  # Will be set later
    }
    
    # Only run simulation test if other checks pass
    if all([results['Python Version'], results['Dependencies'], results['Project Structure']]):
        results['Simulation Test'] = verify_simulation()
    else:
        print("\n⚠️  Skipping simulation test (prerequisites not met)")
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check:.<40} {status}")
    
    print("="*60)
    
    if all(results.values()):
        print("\n✓ All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run full simulation: python safety_twin_simulation.py")
        print("  2. Launch dashboard: streamlit run app.py")
        print("  3. View examples: python example_analysis.py")
        return True
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        
        if not results['Dependencies']:
            print("\nTo install missing dependencies:")
            print("  pip install -r requirements.txt")
        
        return False


def quick_setup():
    """Quick setup helper."""
    print("="*60)
    print("QUICK SETUP")
    print("="*60)
    print()
    
    response = input("Install all required dependencies? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\nInstalling dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                         check=True)
            print("✓ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False
    else:
        print("\nSkipping installation.")
        return False


if __name__ == '__main__':
    print()
    
    # Check if user wants quick setup
    if len(sys.argv) > 1 and sys.argv[1] == '--setup':
        if quick_setup():
            print("\nRunning verification...\n")
            run_full_verification()
    else:
        # Just run verification
        success = run_full_verification()
        
        if not success:
            print("\nTip: Run 'python setup_and_verify.py --setup' for automated installation")
        
        sys.exit(0 if success else 1)

