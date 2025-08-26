"""Project workspace management and tracking system."""

import json
import yaml
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import logging

from ..core.config import get_config_manager
from ..core.logging import get_log_manager


class ProjectStatus(Enum):
    """Project status enumeration."""
    CREATED = "created"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MilestoneStatus(Enum):
    """Milestone status enumeration."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Milestone:
    """Project milestone definition."""
    id: str
    name: str
    description: str
    status: MilestoneStatus
    target_date: Optional[str] = None
    completion_date: Optional[str] = None
    dependencies: List[str] = None
    progress_percentage: float = 0.0
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ProjectMetadata:
    """Project metadata and tracking information."""
    name: str
    description: str
    created_date: str
    last_modified: str
    status: ProjectStatus
    version: str = "1.0.0"
    author: str = "Unknown"
    tags: List[str] = None
    milestones: List[Milestone] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.milestones is None:
            self.milestones = []


class ProjectWorkspace:
    """Project workspace management."""
    
    def __init__(self, workspace_path: Union[str, Path]):
        """Initialize project workspace.
        
        Args:
            workspace_path: Path to the project workspace
        """
        self.workspace_path = Path(workspace_path)
        self.project_file = self.workspace_path / ".fighter_jet_project.json"
        self.config_dir = self.workspace_path / ".fighter_jet"
        self.backup_dir = self.config_dir / "backups"
        self.history_file = self.config_dir / "history.json"
        
        self.logger = get_log_manager().get_logger('project')
        self.metadata: Optional[ProjectMetadata] = None
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_project(self, name: str, description: str, author: str = "Unknown") -> bool:
        """Create a new project workspace.
        
        Args:
            name: Project name
            description: Project description
            author: Project author
            
        Returns:
            True if project created successfully
        """
        try:
            if self.project_file.exists():
                self.logger.warning(f"Project already exists at {self.workspace_path}")
                return False
            
            # Create project metadata
            now = datetime.now(timezone.utc).isoformat()
            self.metadata = ProjectMetadata(
                name=name,
                description=description,
                created_date=now,
                last_modified=now,
                status=ProjectStatus.CREATED,
                author=author
            )
            
            # Create default milestones
            default_milestones = [
                Milestone(
                    id="requirements",
                    name="Requirements Definition",
                    description="Define project requirements and specifications",
                    status=MilestoneStatus.NOT_STARTED
                ),
                Milestone(
                    id="design",
                    name="Design Phase",
                    description="Complete aircraft design and configuration",
                    status=MilestoneStatus.NOT_STARTED,
                    dependencies=["requirements"]
                ),
                Milestone(
                    id="analysis",
                    name="Analysis and Simulation",
                    description="Perform comprehensive analysis and simulations",
                    status=MilestoneStatus.NOT_STARTED,
                    dependencies=["design"]
                ),
                Milestone(
                    id="validation",
                    name="Validation and Testing",
                    description="Validate design against requirements",
                    status=MilestoneStatus.NOT_STARTED,
                    dependencies=["analysis"]
                )
            ]
            
            self.metadata.milestones = default_milestones
            
            # Save project file
            self.save_project()
            
            # Initialize version control if git is available
            self._initialize_version_control()
            
            # Create initial history entry
            self._add_history_entry("project_created", f"Project '{name}' created")
            
            self.logger.info(f"Project '{name}' created successfully at {self.workspace_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            return False
    
    def load_project(self) -> bool:
        """Load existing project from workspace.
        
        Returns:
            True if project loaded successfully
        """
        try:
            if not self.project_file.exists():
                self.logger.error(f"No project found at {self.workspace_path}")
                return False
            
            with open(self.project_file, 'r') as f:
                project_data = json.load(f)
            
            # Convert milestone data
            milestones = []
            for milestone_data in project_data.get('milestones', []):
                milestone_data['status'] = MilestoneStatus(milestone_data['status'])
                milestones.append(Milestone(**milestone_data))
            
            project_data['milestones'] = milestones
            project_data['status'] = ProjectStatus(project_data['status'])
            
            self.metadata = ProjectMetadata(**project_data)
            
            self.logger.info(f"Project '{self.metadata.name}' loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load project: {e}")
            return False
    
    def save_project(self) -> bool:
        """Save project metadata to file.
        
        Returns:
            True if project saved successfully
        """
        try:
            if not self.metadata:
                self.logger.error("No project metadata to save")
                return False
            
            # Update last modified timestamp
            self.metadata.last_modified = datetime.now(timezone.utc).isoformat()
            
            # Convert to dictionary for JSON serialization
            project_data = asdict(self.metadata)
            
            # Convert enums to strings
            project_data['status'] = self.metadata.status.value
            for milestone_data in project_data['milestones']:
                milestone_data['status'] = milestone_data['status'].value
            
            # Save to file
            with open(self.project_file, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            self.logger.debug(f"Project saved to {self.project_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save project: {e}")
            return False
    
    def update_milestone(self, milestone_id: str, status: Optional[MilestoneStatus] = None,
                        progress: Optional[float] = None) -> bool:
        """Update milestone status and progress.
        
        Args:
            milestone_id: Milestone ID to update
            status: New milestone status
            progress: Progress percentage (0-100)
            
        Returns:
            True if milestone updated successfully
        """
        try:
            if not self.metadata:
                self.logger.error("No project loaded")
                return False
            
            # Find milestone
            milestone = None
            for m in self.metadata.milestones:
                if m.id == milestone_id:
                    milestone = m
                    break
            
            if not milestone:
                self.logger.error(f"Milestone '{milestone_id}' not found")
                return False
            
            # Update milestone
            if status:
                old_status = milestone.status
                milestone.status = status
                
                if status == MilestoneStatus.COMPLETED:
                    milestone.completion_date = datetime.now(timezone.utc).isoformat()
                    milestone.progress_percentage = 100.0
                
                self._add_history_entry(
                    "milestone_updated",
                    f"Milestone '{milestone.name}' status changed from {old_status.value} to {status.value}"
                )
            
            if progress is not None:
                milestone.progress_percentage = max(0.0, min(100.0, progress))
                
                if progress >= 100.0 and milestone.status != MilestoneStatus.COMPLETED:
                    milestone.status = MilestoneStatus.COMPLETED
                    milestone.completion_date = datetime.now(timezone.utc).isoformat()
            
            # Save project
            self.save_project()
            
            self.logger.info(f"Milestone '{milestone_id}' updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update milestone: {e}")
            return False
    
    def get_project_status(self) -> Dict[str, Any]:
        """Get comprehensive project status.
        
        Returns:
            Dictionary with project status information
        """
        if not self.metadata:
            return {"error": "No project loaded"}
        
        # Calculate overall progress
        total_milestones = len(self.metadata.milestones)
        completed_milestones = sum(1 for m in self.metadata.milestones 
                                 if m.status == MilestoneStatus.COMPLETED)
        overall_progress = (completed_milestones / total_milestones * 100) if total_milestones > 0 else 0
        
        # Get milestone summary
        milestone_summary = []
        for milestone in self.metadata.milestones:
            milestone_summary.append({
                'id': milestone.id,
                'name': milestone.name,
                'status': milestone.status.value,
                'progress': milestone.progress_percentage,
                'target_date': milestone.target_date,
                'completion_date': milestone.completion_date,
                'dependencies': milestone.dependencies
            })
        
        return {
            'name': self.metadata.name,
            'description': self.metadata.description,
            'status': self.metadata.status.value,
            'version': self.metadata.version,
            'author': self.metadata.author,
            'created_date': self.metadata.created_date,
            'last_modified': self.metadata.last_modified,
            'tags': self.metadata.tags,
            'overall_progress': overall_progress,
            'milestones': milestone_summary,
            'workspace_path': str(self.workspace_path)
        }
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of the current project state.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Path to the created backup
        """
        try:
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"
            
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)
            
            # Copy project file
            if self.project_file.exists():
                shutil.copy2(self.project_file, backup_path / self.project_file.name)
            
            # Copy configuration files
            config_files = [
                "aircraft_configurations",
                "analysis_results",
                "simulation_data"
            ]
            
            for config_file in config_files:
                config_path = self.workspace_path / config_file
                if config_path.exists():
                    if config_path.is_dir():
                        shutil.copytree(config_path, backup_path / config_file, dirs_exist_ok=True)
                    else:
                        shutil.copy2(config_path, backup_path / config_file)
            
            # Create backup metadata
            backup_metadata = {
                'backup_name': backup_name,
                'created_date': datetime.now(timezone.utc).isoformat(),
                'project_name': self.metadata.name if self.metadata else "Unknown",
                'project_version': self.metadata.version if self.metadata else "Unknown"
            }
            
            with open(backup_path / "backup_info.json", 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            self._add_history_entry("backup_created", f"Backup '{backup_name}' created")
            
            self.logger.info(f"Backup '{backup_name}' created at {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore project from backup.
        
        Args:
            backup_name: Name of backup to restore
            
        Returns:
            True if backup restored successfully
        """
        try:
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                self.logger.error(f"Backup '{backup_name}' not found")
                return False
            
            # Create current state backup before restore
            current_backup = self.create_backup("pre_restore_backup")
            
            # Restore project file
            backup_project_file = backup_path / self.project_file.name
            if backup_project_file.exists():
                shutil.copy2(backup_project_file, self.project_file)
            
            # Restore configuration files
            for item in backup_path.iterdir():
                if item.name not in [self.project_file.name, "backup_info.json"]:
                    target_path = self.workspace_path / item.name
                    if item.is_dir():
                        if target_path.exists():
                            shutil.rmtree(target_path)
                        shutil.copytree(item, target_path)
                    else:
                        shutil.copy2(item, target_path)
            
            # Reload project
            self.load_project()
            
            self._add_history_entry("backup_restored", f"Backup '{backup_name}' restored")
            
            self.logger.info(f"Backup '{backup_name}' restored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        try:
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    backup_info_file = backup_dir / "backup_info.json"
                    if backup_info_file.exists():
                        with open(backup_info_file, 'r') as f:
                            backup_info = json.load(f)
                        backups.append(backup_info)
                    else:
                        # Create basic info for backups without metadata
                        backups.append({
                            'backup_name': backup_dir.name,
                            'created_date': datetime.fromtimestamp(backup_dir.stat().st_mtime).isoformat(),
                            'project_name': "Unknown",
                            'project_version': "Unknown"
                        })
        
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
        
        return sorted(backups, key=lambda x: x['created_date'], reverse=True)
    
    def get_project_history(self) -> List[Dict[str, Any]]:
        """Get project history.
        
        Returns:
            List of history entries
        """
        try:
            if not self.history_file.exists():
                return []
            
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            return history.get('entries', [])
            
        except Exception as e:
            self.logger.error(f"Failed to get project history: {e}")
            return []
    
    def _add_history_entry(self, action: str, description: str, metadata: Optional[Dict] = None):
        """Add entry to project history.
        
        Args:
            action: Action type
            description: Action description
            metadata: Optional additional metadata
        """
        try:
            history_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': action,
                'description': description,
                'metadata': metadata or {}
            }
            
            # Load existing history
            history = {'entries': []}
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            
            # Add new entry
            history['entries'].append(history_entry)
            
            # Keep only last 1000 entries
            history['entries'] = history['entries'][-1000:]
            
            # Save history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to add history entry: {e}")
    
    def _initialize_version_control(self):
        """Initialize git repository if git is available."""
        try:
            # Check if git is available
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, cwd=self.workspace_path)
            
            if result.returncode == 0:
                # Initialize git repository
                subprocess.run(['git', 'init'], 
                             capture_output=True, cwd=self.workspace_path)
                
                # Create .gitignore
                gitignore_content = """
# Fighter Jet SDK
.fighter_jet/backups/
*.tmp
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
                gitignore_path = self.workspace_path / '.gitignore'
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content.strip())
                
                # Initial commit
                subprocess.run(['git', 'add', '.'], 
                             capture_output=True, cwd=self.workspace_path)
                subprocess.run(['git', 'commit', '-m', 'Initial project setup'], 
                             capture_output=True, cwd=self.workspace_path)
                
                self.logger.info("Git repository initialized")
                
        except Exception as e:
            self.logger.debug(f"Git initialization failed (optional): {e}")


class ProjectManager:
    """High-level project management interface."""
    
    def __init__(self):
        """Initialize project manager."""
        self.logger = get_log_manager().get_logger('project_manager')
        self.current_workspace: Optional[ProjectWorkspace] = None
    
    def create_project(self, workspace_path: str, name: str, description: str, 
                      author: str = "Unknown") -> bool:
        """Create a new project.
        
        Args:
            workspace_path: Path to create project workspace
            name: Project name
            description: Project description
            author: Project author
            
        Returns:
            True if project created successfully
        """
        try:
            workspace = ProjectWorkspace(workspace_path)
            if workspace.create_project(name, description, author):
                self.current_workspace = workspace
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to create project: {e}")
            return False
    
    def open_project(self, workspace_path: str) -> bool:
        """Open existing project.
        
        Args:
            workspace_path: Path to project workspace
            
        Returns:
            True if project opened successfully
        """
        try:
            workspace = ProjectWorkspace(workspace_path)
            if workspace.load_project():
                self.current_workspace = workspace
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to open project: {e}")
            return False
    
    def get_current_project_status(self) -> Optional[Dict[str, Any]]:
        """Get status of current project.
        
        Returns:
            Project status dictionary or None if no project open
        """
        if not self.current_workspace:
            return None
        
        return self.current_workspace.get_project_status()
    
    def update_milestone(self, milestone_id: str, status: Optional[str] = None,
                        progress: Optional[float] = None) -> bool:
        """Update milestone in current project.
        
        Args:
            milestone_id: Milestone ID
            status: New status string
            progress: Progress percentage
            
        Returns:
            True if milestone updated successfully
        """
        if not self.current_workspace:
            self.logger.error("No project currently open")
            return False
        
        milestone_status = None
        if status:
            try:
                milestone_status = MilestoneStatus(status)
            except ValueError:
                self.logger.error(f"Invalid milestone status: {status}")
                return False
        
        return self.current_workspace.update_milestone(milestone_id, milestone_status, progress)
    
    def create_backup(self, backup_name: Optional[str] = None) -> Optional[str]:
        """Create backup of current project.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Path to backup or None if failed
        """
        if not self.current_workspace:
            self.logger.error("No project currently open")
            return None
        
        try:
            return self.current_workspace.create_backup(backup_name)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List backups for current project.
        
        Returns:
            List of backup information
        """
        if not self.current_workspace:
            return []
        
        return self.current_workspace.list_backups()
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore backup for current project.
        
        Args:
            backup_name: Name of backup to restore
            
        Returns:
            True if backup restored successfully
        """
        if not self.current_workspace:
            self.logger.error("No project currently open")
            return False
        
        return self.current_workspace.restore_backup(backup_name)
    
    def get_project_history(self) -> List[Dict[str, Any]]:
        """Get history for current project.
        
        Returns:
            List of history entries
        """
        if not self.current_workspace:
            return []
        
        return self.current_workspace.get_project_history()


# Global project manager instance
project_manager = ProjectManager()