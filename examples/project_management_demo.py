#!/usr/bin/env python3
"""
Demonstration of the Fighter Jet SDK Project Management System.

This script shows how to use the project management functionality
to create, manage, and track fighter jet design projects.
"""

import tempfile
import shutil
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fighter_jet_sdk.cli.project_manager import ProjectManager, MilestoneStatus


def main():
    """Demonstrate project management functionality."""
    print("=== Fighter Jet SDK Project Management Demo ===\n")
    
    # Create temporary workspace for demo
    temp_dir = tempfile.mkdtemp()
    workspace_path = Path(temp_dir) / "demo_fighter_jet_project"
    workspace_path.mkdir()
    
    try:
        # Initialize project manager
        manager = ProjectManager()
        
        # 1. Create a new project
        print("1. Creating new fighter jet project...")
        success = manager.create_project(
            str(workspace_path),
            "Advanced Stealth Fighter",
            "Next-generation stealth fighter with modular design capabilities",
            "Demo Engineer"
        )
        
        if success:
            print("✓ Project created successfully!")
        else:
            print("✗ Failed to create project")
            return
        
        # 2. Show initial project status
        print("\n2. Initial project status:")
        status = manager.get_current_project_status()
        print(f"   Name: {status['name']}")
        print(f"   Description: {status['description']}")
        print(f"   Status: {status['status']}")
        print(f"   Progress: {status['overall_progress']:.1f}%")
        print(f"   Milestones: {len(status['milestones'])}")
        
        # 3. Update milestones to show progress
        print("\n3. Updating project milestones...")
        
        # Complete requirements phase
        manager.update_milestone("requirements", "completed", 100.0)
        print("   ✓ Requirements phase completed")
        
        # Start design phase
        manager.update_milestone("design", "in_progress", 60.0)
        print("   ✓ Design phase in progress (60%)")
        
        # Start analysis phase
        manager.update_milestone("analysis", "in_progress", 25.0)
        print("   ✓ Analysis phase started (25%)")
        
        # 4. Show updated project status
        print("\n4. Updated project status:")
        status = manager.get_current_project_status()
        print(f"   Overall Progress: {status['overall_progress']:.1f}%")
        
        print("   Milestone Details:")
        for milestone in status['milestones']:
            status_icon = {
                'not_started': '⚪',
                'in_progress': '🟡',
                'completed': '✅',
                'blocked': '🔴'
            }.get(milestone['status'], '❓')
            
            print(f"   {status_icon} {milestone['name']}: {milestone['progress']:.1f}%")
            if milestone['dependencies']:
                print(f"      Dependencies: {', '.join(milestone['dependencies'])}")
        
        # 5. Create project backup
        print("\n5. Creating project backup...")
        backup_path = manager.create_backup("milestone_progress_backup")
        if backup_path:
            print(f"   ✓ Backup created: {Path(backup_path).name}")
        
        # 6. List available backups
        print("\n6. Available backups:")
        backups = manager.list_backups()
        for backup in backups:
            print(f"   📦 {backup['backup_name']}")
            print(f"      Created: {backup['created_date']}")
            print(f"      Project: {backup['project_name']} v{backup['project_version']}")
        
        # 7. Show project history
        print("\n7. Project history (last 5 entries):")
        history = manager.get_project_history()
        for entry in history[-5:]:
            timestamp = entry['timestamp'][:19].replace('T', ' ')
            print(f"   [{timestamp}] {entry['action']}: {entry['description']}")
        
        # 8. Simulate project completion
        print("\n8. Completing remaining milestones...")
        manager.update_milestone("design", "completed", 100.0)
        manager.update_milestone("analysis", "completed", 100.0)
        manager.update_milestone("validation", "completed", 100.0)
        
        # 9. Final project status
        print("\n9. Final project status:")
        status = manager.get_current_project_status()
        print(f"   Overall Progress: {status['overall_progress']:.1f}%")
        print("   All milestones completed! 🎉")
        
        # 10. Create final backup
        print("\n10. Creating final project backup...")
        final_backup = manager.create_backup("project_completed_backup")
        if final_backup:
            print(f"    ✓ Final backup created: {Path(final_backup).name}")
        
        print("\n=== Demo completed successfully! ===")
        print(f"Demo workspace: {workspace_path}")
        print("Note: This is a temporary workspace that will be cleaned up.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        
    finally:
        # Clean up temporary workspace
        print(f"\nCleaning up temporary workspace...")
        shutil.rmtree(temp_dir)
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()