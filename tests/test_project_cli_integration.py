"""Tests for project management CLI integration."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

from fighter_jet_sdk.cli.main import handle_project_command, create_cli


class TestProjectCLIIntegration:
    """Test CLI integration for project management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir) / "test_project"
        self.workspace_path.mkdir()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_project_command(self):
        """Test project create command."""
        # Create mock args
        args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        
        result = handle_project_command(args)
        
        assert result == 0
        assert (self.workspace_path / '.fighter_jet_project.json').exists()
    
    def test_create_project_command_no_author(self):
        """Test project create command without author."""
        args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author=None,
            path=str(self.workspace_path)
        )
        
        result = handle_project_command(args)
        
        assert result == 0
        assert (self.workspace_path / '.fighter_jet_project.json').exists()
    
    def test_create_project_command_no_path(self):
        """Test project create command without path."""
        args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=None
        )
        
        with patch('os.getcwd', return_value=str(self.workspace_path)):
            result = handle_project_command(args)
        
        assert result == 0
        assert (self.workspace_path / '.fighter_jet_project.json').exists()
    
    def test_open_project_command(self):
        """Test project open command."""
        # Create project first
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Open project
        open_args = argparse.Namespace(
            project_action='open',
            path=str(self.workspace_path)
        )
        
        result = handle_project_command(open_args)
        assert result == 0
    
    def test_open_project_command_no_path(self):
        """Test project open command without path."""
        # Create project first
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Open project without path
        open_args = argparse.Namespace(
            project_action='open',
            path=None
        )
        
        with patch('os.getcwd', return_value=str(self.workspace_path)):
            result = handle_project_command(open_args)
        
        assert result == 0
    
    def test_open_project_command_not_exists(self):
        """Test opening non-existent project."""
        args = argparse.Namespace(
            project_action='open',
            path=str(self.workspace_path)
        )
        
        result = handle_project_command(args)
        assert result == 1
    
    def test_status_command_no_project(self):
        """Test status command when no project is open."""
        args = argparse.Namespace(project_action='status')
        
        result = handle_project_command(args)
        assert result == 1
    
    def test_status_command_with_project(self):
        """Test status command with open project."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Get status
        status_args = argparse.Namespace(project_action='status')
        result = handle_project_command(status_args)
        
        assert result == 0
    
    def test_milestone_command(self):
        """Test milestone update command."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Update milestone
        milestone_args = argparse.Namespace(
            project_action='milestone',
            id='requirements',
            status='in_progress',
            progress=50.0
        )
        
        result = handle_project_command(milestone_args)
        assert result == 0
    
    def test_milestone_command_invalid(self):
        """Test milestone update with invalid milestone."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Update invalid milestone
        milestone_args = argparse.Namespace(
            project_action='milestone',
            id='nonexistent',
            status='in_progress',
            progress=50.0
        )
        
        result = handle_project_command(milestone_args)
        assert result == 1
    
    def test_backup_command(self):
        """Test backup creation command."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Create backup
        backup_args = argparse.Namespace(
            project_action='backup',
            name='test_backup'
        )
        
        result = handle_project_command(backup_args)
        assert result == 0
    
    def test_backup_command_no_name(self):
        """Test backup creation without name."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Create backup without name
        backup_args = argparse.Namespace(
            project_action='backup',
            name=None
        )
        
        result = handle_project_command(backup_args)
        assert result == 0
    
    def test_list_backups_command(self):
        """Test list backups command."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Create backup
        backup_args = argparse.Namespace(
            project_action='backup',
            name='test_backup'
        )
        handle_project_command(backup_args)
        
        # List backups
        list_args = argparse.Namespace(project_action='list-backups')
        result = handle_project_command(list_args)
        
        assert result == 0
    
    def test_list_backups_command_empty(self):
        """Test list backups command with no backups."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # List backups (should be empty)
        list_args = argparse.Namespace(project_action='list-backups')
        result = handle_project_command(list_args)
        
        assert result == 0
    
    def test_restore_backup_command(self):
        """Test restore backup command."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Create backup
        backup_args = argparse.Namespace(
            project_action='backup',
            name='test_backup'
        )
        handle_project_command(backup_args)
        
        # Restore backup
        restore_args = argparse.Namespace(
            project_action='restore',
            backup='test_backup'
        )
        
        result = handle_project_command(restore_args)
        assert result == 0
    
    def test_restore_backup_command_not_found(self):
        """Test restore backup command with non-existent backup."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Restore non-existent backup
        restore_args = argparse.Namespace(
            project_action='restore',
            backup='nonexistent_backup'
        )
        
        result = handle_project_command(restore_args)
        assert result == 1
    
    def test_history_command(self):
        """Test project history command."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Get history
        history_args = argparse.Namespace(
            project_action='history',
            limit=10
        )
        
        result = handle_project_command(history_args)
        assert result == 0
    
    def test_history_command_empty(self):
        """Test history command with no history."""
        # Create and open project
        create_args = argparse.Namespace(
            project_action='create',
            name='Test Project',
            description='A test project',
            author='Test Author',
            path=str(self.workspace_path)
        )
        handle_project_command(create_args)
        
        # Clear history by creating new workspace
        from fighter_jet_sdk.cli.project_manager import project_manager
        project_manager.current_workspace = None
        
        # Get history (should be empty)
        history_args = argparse.Namespace(
            project_action='history',
            limit=10
        )
        
        result = handle_project_command(history_args)
        assert result == 0
    
    def test_unknown_action(self):
        """Test unknown project action."""
        args = argparse.Namespace(project_action='unknown_action')
        
        result = handle_project_command(args)
        assert result == 1
    
    def test_cli_parser_project_commands(self):
        """Test that CLI parser includes all project commands."""
        parser = create_cli()
        
        # Test project create command
        args = parser.parse_args([
            'project', 'create',
            '--name', 'Test Project',
            '--description', 'Test Description',
            '--author', 'Test Author',
            '--path', '/test/path'
        ])
        
        assert args.command == 'project'
        assert args.project_action == 'create'
        assert args.name == 'Test Project'
        assert args.description == 'Test Description'
        assert args.author == 'Test Author'
        assert args.path == '/test/path'
        
        # Test project open command
        args = parser.parse_args(['project', 'open', '--path', '/test/path'])
        assert args.project_action == 'open'
        assert args.path == '/test/path'
        
        # Test project status command
        args = parser.parse_args(['project', 'status'])
        assert args.project_action == 'status'
        
        # Test project milestone command
        args = parser.parse_args([
            'project', 'milestone',
            '--id', 'requirements',
            '--status', 'in_progress',
            '--progress', '50.0'
        ])
        assert args.project_action == 'milestone'
        assert args.id == 'requirements'
        assert args.status == 'in_progress'
        assert args.progress == 50.0
        
        # Test project backup command
        args = parser.parse_args(['project', 'backup', '--name', 'test_backup'])
        assert args.project_action == 'backup'
        assert args.name == 'test_backup'
        
        # Test project list-backups command
        args = parser.parse_args(['project', 'list-backups'])
        assert args.project_action == 'list-backups'
        
        # Test project restore command
        args = parser.parse_args(['project', 'restore', '--backup', 'test_backup'])
        assert args.project_action == 'restore'
        assert args.backup == 'test_backup'
        
        # Test project history command
        args = parser.parse_args(['project', 'history', '--limit', '10'])
        assert args.project_action == 'history'
        assert args.limit == 10


if __name__ == '__main__':
    pytest.main([__file__])