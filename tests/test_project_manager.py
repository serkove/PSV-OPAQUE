"""Tests for project management functionality."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from fighter_jet_sdk.cli.project_manager import (
    ProjectWorkspace, ProjectManager, ProjectMetadata, Milestone,
    ProjectStatus, MilestoneStatus, project_manager
)


class TestProjectWorkspace:
    """Test ProjectWorkspace functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "test_project"
        self.workspace_path.mkdir()
        self.workspace = ProjectWorkspace(self.workspace_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_project(self):
        """Test project creation."""
        result = self.workspace.create_project(
            name="Test Fighter Jet",
            description="A test fighter jet project",
            author="Test Engineer"
        )
        
        assert result is True
        assert self.workspace.project_file.exists()
        assert self.workspace.config_dir.exists()
        assert self.workspace.backup_dir.exists()
        
        # Check project metadata
        assert self.workspace.metadata is not None
        assert self.workspace.metadata.name == "Test Fighter Jet"
        assert self.workspace.metadata.description == "A test fighter jet project"
        assert self.workspace.metadata.author == "Test Engineer"
        assert self.workspace.metadata.status == ProjectStatus.CREATED
        
        # Check default milestones
        assert len(self.workspace.metadata.milestones) == 4
        milestone_ids = [m.id for m in self.workspace.metadata.milestones]
        assert "requirements" in milestone_ids
        assert "design" in milestone_ids
        assert "analysis" in milestone_ids
        assert "validation" in milestone_ids
    
    def test_create_project_already_exists(self):
        """Test creating project when one already exists."""
        # Create first project
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Try to create another project in same location
        result = self.workspace.create_project("Another Project", "Description", "Author")
        assert result is False
    
    def test_load_project(self):
        """Test loading existing project."""
        # Create project first
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Create new workspace instance and load
        new_workspace = ProjectWorkspace(self.workspace_path)
        result = new_workspace.load_project()
        
        assert result is True
        assert new_workspace.metadata is not None
        assert new_workspace.metadata.name == "Test Project"
        assert new_workspace.metadata.description == "Description"
        assert new_workspace.metadata.author == "Author"
    
    def test_load_project_not_exists(self):
        """Test loading project that doesn't exist."""
        result = self.workspace.load_project()
        assert result is False
    
    def test_save_project(self):
        """Test saving project metadata."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Modify metadata
        self.workspace.metadata.description = "Updated description"
        
        # Save project
        result = self.workspace.save_project()
        assert result is True
        
        # Load and verify changes
        new_workspace = ProjectWorkspace(self.workspace_path)
        new_workspace.load_project()
        assert new_workspace.metadata.description == "Updated description"
    
    def test_update_milestone(self):
        """Test updating milestone status and progress."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Update milestone status
        result = self.workspace.update_milestone(
            "requirements", 
            status=MilestoneStatus.IN_PROGRESS,
            progress=50.0
        )
        
        assert result is True
        
        # Find updated milestone
        requirements_milestone = None
        for milestone in self.workspace.metadata.milestones:
            if milestone.id == "requirements":
                requirements_milestone = milestone
                break
        
        assert requirements_milestone is not None
        assert requirements_milestone.status == MilestoneStatus.IN_PROGRESS
        assert requirements_milestone.progress_percentage == 50.0
    
    def test_update_milestone_complete(self):
        """Test completing a milestone."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Complete milestone
        result = self.workspace.update_milestone(
            "requirements",
            status=MilestoneStatus.COMPLETED
        )
        
        assert result is True
        
        # Check milestone
        requirements_milestone = None
        for milestone in self.workspace.metadata.milestones:
            if milestone.id == "requirements":
                requirements_milestone = milestone
                break
        
        assert requirements_milestone.status == MilestoneStatus.COMPLETED
        assert requirements_milestone.progress_percentage == 100.0
        assert requirements_milestone.completion_date is not None
    
    def test_update_milestone_not_found(self):
        """Test updating non-existent milestone."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        result = self.workspace.update_milestone("nonexistent", MilestoneStatus.COMPLETED)
        assert result is False
    
    def test_get_project_status(self):
        """Test getting comprehensive project status."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Update some milestones
        self.workspace.update_milestone("requirements", MilestoneStatus.COMPLETED)
        self.workspace.update_milestone("design", MilestoneStatus.IN_PROGRESS, 30.0)
        
        status = self.workspace.get_project_status()
        
        assert status["name"] == "Test Project"
        assert status["description"] == "Description"
        assert status["author"] == "Author"
        assert status["status"] == "created"
        assert status["overall_progress"] == 25.0  # 1 of 4 milestones completed
        assert len(status["milestones"]) == 4
        
        # Check specific milestone status
        requirements_status = None
        for milestone in status["milestones"]:
            if milestone["id"] == "requirements":
                requirements_status = milestone
                break
        
        assert requirements_status["status"] == "completed"
        assert requirements_status["progress"] == 100.0
    
    def test_create_backup(self):
        """Test creating project backup."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Create some test files
        test_file = self.workspace_path / "test_config.json"
        test_file.write_text('{"test": "data"}')
        
        # Create backup
        backup_path = self.workspace.create_backup("test_backup")
        
        assert backup_path is not None
        backup_dir = Path(backup_path)
        assert backup_dir.exists()
        assert (backup_dir / self.workspace.project_file.name).exists()
        assert (backup_dir / "backup_info.json").exists()
    
    def test_list_backups(self):
        """Test listing available backups."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Create multiple backups
        self.workspace.create_backup("backup1")
        self.workspace.create_backup("backup2")
        
        backups = self.workspace.list_backups()
        
        assert len(backups) == 2
        backup_names = [b["backup_name"] for b in backups]
        assert "backup1" in backup_names
        assert "backup2" in backup_names
    
    def test_restore_backup(self):
        """Test restoring from backup."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Create backup
        self.workspace.create_backup("test_backup")
        
        # Modify project
        self.workspace.metadata.description = "Modified description"
        self.workspace.save_project()
        
        # Restore backup
        result = self.workspace.restore_backup("test_backup")
        assert result is True
        
        # Verify restoration
        assert self.workspace.metadata.description == "Description"
    
    def test_restore_backup_not_found(self):
        """Test restoring non-existent backup."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        result = self.workspace.restore_backup("nonexistent_backup")
        assert result is False
    
    def test_get_project_history(self):
        """Test getting project history."""
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Perform some actions to create history
        self.workspace.update_milestone("requirements", MilestoneStatus.IN_PROGRESS)
        self.workspace.create_backup("test_backup")
        
        history = self.workspace.get_project_history()
        
        assert len(history) >= 3  # project_created, milestone_updated, backup_created
        
        # Check history entry structure
        for entry in history:
            assert "timestamp" in entry
            assert "action" in entry
            assert "description" in entry
            assert "metadata" in entry
    
    @patch('subprocess.run')
    def test_initialize_version_control_success(self, mock_run):
        """Test successful git initialization."""
        # Mock successful git commands
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git --version
            MagicMock(returncode=0),  # git init
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=0),  # git commit
        ]
        
        self.workspace.create_project("Test Project", "Description", "Author")
        
        # Verify git commands were called
        assert mock_run.call_count == 4
        
        # Check .gitignore was created
        gitignore_path = self.workspace_path / '.gitignore'
        assert gitignore_path.exists()
    
    @patch('subprocess.run')
    def test_initialize_version_control_failure(self, mock_run):
        """Test git initialization failure."""
        # Mock git not available
        mock_run.side_effect = [
            MagicMock(returncode=1),  # git --version fails
        ]
        
        # Should not raise exception
        self.workspace.create_project("Test Project", "Description", "Author")
        
        assert mock_run.call_count == 1


class TestProjectManager:
    """Test ProjectManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "test_project"
        self.workspace_path.mkdir()
        self.manager = ProjectManager()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_project(self):
        """Test creating project through manager."""
        result = self.manager.create_project(
            str(self.workspace_path),
            "Test Project",
            "Description",
            "Author"
        )
        
        assert result is True
        assert self.manager.current_workspace is not None
        assert self.manager.current_workspace.metadata.name == "Test Project"
    
    def test_open_project(self):
        """Test opening existing project through manager."""
        # Create project first
        workspace = ProjectWorkspace(self.workspace_path)
        workspace.create_project("Test Project", "Description", "Author")
        
        # Open through manager
        result = self.manager.open_project(str(self.workspace_path))
        
        assert result is True
        assert self.manager.current_workspace is not None
        assert self.manager.current_workspace.metadata.name == "Test Project"
    
    def test_get_current_project_status_no_project(self):
        """Test getting status when no project is open."""
        status = self.manager.get_current_project_status()
        assert status is None
    
    def test_get_current_project_status(self):
        """Test getting current project status."""
        self.manager.create_project(
            str(self.workspace_path),
            "Test Project",
            "Description",
            "Author"
        )
        
        status = self.manager.get_current_project_status()
        
        assert status is not None
        assert status["name"] == "Test Project"
        assert status["description"] == "Description"
        assert status["author"] == "Author"
    
    def test_update_milestone(self):
        """Test updating milestone through manager."""
        self.manager.create_project(
            str(self.workspace_path),
            "Test Project",
            "Description",
            "Author"
        )
        
        result = self.manager.update_milestone("requirements", "in_progress", 50.0)
        assert result is True
    
    def test_update_milestone_no_project(self):
        """Test updating milestone when no project is open."""
        result = self.manager.update_milestone("requirements", "in_progress")
        assert result is False
    
    def test_update_milestone_invalid_status(self):
        """Test updating milestone with invalid status."""
        self.manager.create_project(
            str(self.workspace_path),
            "Test Project",
            "Description",
            "Author"
        )
        
        result = self.manager.update_milestone("requirements", "invalid_status")
        assert result is False
    
    def test_create_backup(self):
        """Test creating backup through manager."""
        self.manager.create_project(
            str(self.workspace_path),
            "Test Project",
            "Description",
            "Author"
        )
        
        backup_path = self.manager.create_backup("test_backup")
        assert backup_path is not None
        assert Path(backup_path).exists()
    
    def test_create_backup_no_project(self):
        """Test creating backup when no project is open."""
        backup_path = self.manager.create_backup("test_backup")
        assert backup_path is None
    
    def test_list_backups(self):
        """Test listing backups through manager."""
        self.manager.create_project(
            str(self.workspace_path),
            "Test Project",
            "Description",
            "Author"
        )
        
        self.manager.create_backup("backup1")
        self.manager.create_backup("backup2")
        
        backups = self.manager.list_backups()
        assert len(backups) == 2
    
    def test_list_backups_no_project(self):
        """Test listing backups when no project is open."""
        backups = self.manager.list_backups()
        assert backups == []
    
    def test_restore_backup(self):
        """Test restoring backup through manager."""
        self.manager.create_project(
            str(self.workspace_path),
            "Test Project",
            "Description",
            "Author"
        )
        
        self.manager.create_backup("test_backup")
        
        # Modify project
        self.manager.current_workspace.metadata.description = "Modified"
        self.manager.current_workspace.save_project()
        
        # Restore
        result = self.manager.restore_backup("test_backup")
        assert result is True
        assert self.manager.current_workspace.metadata.description == "Description"
    
    def test_restore_backup_no_project(self):
        """Test restoring backup when no project is open."""
        result = self.manager.restore_backup("test_backup")
        assert result is False
    
    def test_get_project_history(self):
        """Test getting project history through manager."""
        self.manager.create_project(
            str(self.workspace_path),
            "Test Project",
            "Description",
            "Author"
        )
        
        history = self.manager.get_project_history()
        assert len(history) >= 1  # At least project creation
    
    def test_get_project_history_no_project(self):
        """Test getting history when no project is open."""
        history = self.manager.get_project_history()
        assert history == []


class TestMilestone:
    """Test Milestone data class."""
    
    def test_milestone_creation(self):
        """Test creating milestone."""
        milestone = Milestone(
            id="test",
            name="Test Milestone",
            description="A test milestone",
            status=MilestoneStatus.NOT_STARTED
        )
        
        assert milestone.id == "test"
        assert milestone.name == "Test Milestone"
        assert milestone.description == "A test milestone"
        assert milestone.status == MilestoneStatus.NOT_STARTED
        assert milestone.dependencies == []
        assert milestone.progress_percentage == 0.0
    
    def test_milestone_with_dependencies(self):
        """Test creating milestone with dependencies."""
        milestone = Milestone(
            id="test",
            name="Test Milestone",
            description="A test milestone",
            status=MilestoneStatus.NOT_STARTED,
            dependencies=["req1", "req2"]
        )
        
        assert milestone.dependencies == ["req1", "req2"]


class TestProjectMetadata:
    """Test ProjectMetadata data class."""
    
    def test_project_metadata_creation(self):
        """Test creating project metadata."""
        now = datetime.now(timezone.utc).isoformat()
        metadata = ProjectMetadata(
            name="Test Project",
            description="A test project",
            created_date=now,
            last_modified=now,
            status=ProjectStatus.CREATED
        )
        
        assert metadata.name == "Test Project"
        assert metadata.description == "A test project"
        assert metadata.status == ProjectStatus.CREATED
        assert metadata.tags == []
        assert metadata.milestones == []
        assert metadata.version == "1.0.0"
        assert metadata.author == "Unknown"


class TestGlobalProjectManager:
    """Test global project manager instance."""
    
    def test_global_instance(self):
        """Test that global project manager instance exists."""
        assert project_manager is not None
        assert isinstance(project_manager, ProjectManager)


if __name__ == '__main__':
    pytest.main([__file__])