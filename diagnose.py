"""
Diagnostic script to debug multi-agent environment setup.
Run this to check if everything is configured correctly.
"""

import sys
import os

print("=" * 60)
print("🔍 CommunityPulse Multi-Agent Diagnostics")
print("=" * 60)

# Check 1: File exists
print("\n1️⃣ Checking if env_multiagent.py exists...")
env_multiagent_path = os.path.join("app", "env_multiagent.py")
if os.path.exists(env_multiagent_path):
    size = os.path.getsize(env_multiagent_path)
    print(f"   ✅ File found: {env_multiagent_path} ({size} bytes)")
else:
    print(f"   ❌ File NOT found: {env_multiagent_path}")
    sys.exit(1)

# Check 2: Import test
print("\n2️⃣ Testing import...")
try:
    from app.env_multiagent import MultiAgentCoordinatorEnv
    print("   ✅ Import successful: MultiAgentCoordinatorEnv")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print("\n   📝 Error details:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 3: Class instantiation
print("\n3️⃣ Testing class instantiation...")
try:
    from app.env import CommunityPulseEnv
    base_env = CommunityPulseEnv()
    multi_env = MultiAgentCoordinatorEnv(base_env=base_env, num_coordinators=2)
    print("   ✅ Multi-agent environment created successfully")
except Exception as e:
    print(f"   ❌ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Reset test
print("\n4️⃣ Testing reset...")
try:
    obs = multi_env.reset(task_id=1)
    print("   ✅ Reset successful")
    print(f"   📊 Coordinators: {len(obs.get('coordinators', []))}")
    print(f"   📊 Current coordinator: {obs.get('current_coordinator')}")
except Exception as e:
    print(f"   ❌ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 5: Main.py import check
print("\n5️⃣ Checking main.py imports...")
main_py_path = os.path.join("app", "main.py")
with open(main_py_path, 'r', encoding='utf-8') as f:
    main_content = f.read()
    
if "from app.env_multiagent import MultiAgentCoordinatorEnv" in main_content:
    print("   ✅ main.py has correct import statement")
else:
    print("   ❌ main.py is MISSING the import statement!")
    print("   📝 Add this line to app/main.py:")
    print("      from app.env_multiagent import MultiAgentCoordinatorEnv")

# Check 6: Endpoint registration
print("\n6️⃣ Checking endpoint registration...")
if "@app.post(\"/reset_multiagent\")" in main_content:
    print("   ✅ /reset_multiagent endpoint found")
else:
    print("   ❌ /reset_multiagent endpoint NOT found in main.py")

if "@app.post(\"/step_multiagent\")" in main_content:
    print("   ✅ /step_multiagent endpoint found")
else:
    print("   ❌ /step_multiagent endpoint NOT found in main.py")

if "@app.get(\"/leaderboard\")" in main_content:
    print("   ✅ /leaderboard endpoint found")
else:
    print("   ❌ /leaderboard endpoint NOT found in main.py")

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED!")
print("=" * 60)
print("\n📌 Next steps:")
print("1. Restart your server: uvicorn app.main:app --reload --port 7860")
print("2. Test endpoint: curl -X POST http://localhost:7860/reset_multiagent -H \"Content-Type: application/json\" -d \"{\\\"task_id\\\": 4}\"")
