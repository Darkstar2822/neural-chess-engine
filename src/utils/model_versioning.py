import os
import json
import torch
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from src.neural_network.chess_net import ChessNet

class ModelVersion:
    def __init__(self, version: str, model_path: str, metadata: Dict[str, Any]):
        self.version = version
        self.model_path = model_path
        self.metadata = metadata
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'model_path': self.model_path,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        version = cls(data['version'], data['model_path'], data['metadata'])
        version.created_at = data.get('created_at', datetime.now().isoformat())
        return version

class ModelVersionManager:
    def __init__(self, versions_file: str = "models/versions.json"):
        self.versions_file = versions_file
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, ModelVersion]:
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                return {v: ModelVersion.from_dict(data[v]) for v in data}
            except:
                return {}
        return {}
    
    def _save_versions(self):
        os.makedirs(os.path.dirname(self.versions_file), exist_ok=True)
        with open(self.versions_file, 'w') as f:
            json.dump({v: self.versions[v].to_dict() for v in self.versions}, f, indent=2)
    
    def get_model_hash(self, model_path: str) -> str:
        """Generate hash of model file for version tracking"""
        with open(model_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    
    def register_model(self, model_path: str, metadata: Dict[str, Any]) -> str:
        """Register a new model version"""
        model_hash = self.get_model_hash(model_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{len(self.versions) + 1}_{timestamp}_{model_hash}"
        
        self.versions[version] = ModelVersion(version, model_path, metadata)
        self._save_versions()
        
        return version
    
    def get_latest_version(self) -> Optional[ModelVersion]:
        """Get the most recent model version"""
        if not self.versions:
            return None
        
        latest_key = max(self.versions.keys(), key=lambda v: self.versions[v].created_at)
        return self.versions[latest_key]
    
    def get_compatible_models(self, architecture_version: str) -> list[ModelVersion]:
        """Get models compatible with specific architecture version"""
        compatible = []
        for version in self.versions.values():
            if version.metadata.get('architecture_version') == architecture_version:
                compatible.append(version)
        
        return sorted(compatible, key=lambda v: v.created_at, reverse=True)
    
    def is_compatible(self, model_version: str, current_architecture: str) -> bool:
        """Check if model version is compatible with current architecture"""
        if model_version not in self.versions:
            return False
        
        model = self.versions[model_version]
        return model.metadata.get('architecture_version') == current_architecture
    
    def migrate_model(self, old_version: str, new_architecture: str) -> Optional[str]:
        """Migrate model to new architecture (placeholder for future implementation)"""
        # This would contain logic to convert models between architecture versions
        # For now, just return None to indicate migration not supported
        return None
    
    def cleanup_old_versions(self, keep_count: int = 10):
        """Remove old model versions, keeping only the most recent ones"""
        if len(self.versions) <= keep_count:
            return
        
        sorted_versions = sorted(self.versions.items(), 
                               key=lambda x: x[1].created_at, reverse=True)
        
        to_remove = sorted_versions[keep_count:]
        
        for version_key, version in to_remove:
            # Remove model file
            try:
                if os.path.exists(version.model_path):
                    os.remove(version.model_path)
            except:
                pass
            
            # Remove from versions dict
            del self.versions[version_key]
        
        self._save_versions()
    
    def get_model_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model version"""
        if version not in self.versions:
            return None
        
        model_version = self.versions[version]
        info = model_version.to_dict()
        
        # Add file size if model exists
        if os.path.exists(model_version.model_path):
            info['file_size_mb'] = os.path.getsize(model_version.model_path) / (1024 * 1024)
        
        return info

# Global model version manager
model_version_manager = ModelVersionManager()