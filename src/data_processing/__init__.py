"""
Enhanced Google Drive data access utilities for soil moisture analysis.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import re
from datetime import datetime
import warnings
import requests
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveLoader:
    """
    Enhanced Google Drive loader for Sentinel satellite data with validation and file discovery.
    """
    
    def __init__(self, drive_path: str = "/content/drive/MyDrive/"):
        self.drive_path = Path(drive_path)
        self.s1_dir = self.drive_path / "SOIL_MOISTURE"
        self.s2_dir = self.drive_path / "soil_moisture_s2"
        
        # Supported file extensions
        self.supported_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        
        # File pattern matching for Sentinel data
        self.s1_patterns = [
            r'S1[AB]_.*\.tif{1,2}$',  # Standard Sentinel-1 naming
            r'.*VV.*\.tif{1,2}$',     # VV polarization
            r'.*VH.*\.tif{1,2}$',     # VH polarization
            r'sentinel.*1.*\.tif{1,2}$'  # Generic Sentinel-1
        ]
        
        self.s2_patterns = [
            r'S2[AB]_.*\.tif{1,2}$',  # Standard Sentinel-2 naming
            r'.*B[0-9]{2}.*\.tif{1,2}$',  # Band files (B02, B03, etc.)
            r'sentinel.*2.*\.tif{1,2}$'   # Generic Sentinel-2
        ]
        
        # Initialize and validate
        self._validate_setup()
    
    def _validate_setup(self) -> bool:
        """Validate Google Drive setup and directory structure."""
        if not self._is_drive_mounted():
            logger.error("Google Drive is not mounted! Please run: drive.mount('/content/drive')")
            return False
        
        if not self.drive_path.exists():
            logger.error(f"Drive path does not exist: {self.drive_path}")
            return False
        
        # Check and create directories if needed
        missing_dirs = []
        if not self.s1_dir.exists():
            missing_dirs.append(str(self.s1_dir))
        if not self.s2_dir.exists():
            missing_dirs.append(str(self.s2_dir))
        
        if missing_dirs:
            logger.warning(f"Missing directories: {missing_dirs}")
            logger.info("You can create them or update the paths using set_custom_paths()")
        
        return True
    
    def _is_drive_mounted(self) -> bool:
        """Check if Google Drive is properly mounted."""
        return Path("/content/drive").exists()
    
    def mount_drive(self):
        """Mount Google Drive if not already mounted."""
        if not self._is_drive_mounted():
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                logger.info("Google Drive mounted successfully!")
                self._validate_setup()
            except ImportError:
                logger.error("Not running in Google Colab environment")
            except Exception as e:
                logger.error(f"Failed to mount Google Drive: {e}")
        else:
            logger.info("Google Drive already mounted")
    
    def set_custom_paths(self, s1_dir: Optional[str] = None, s2_dir: Optional[str] = None):
        """Set custom directory paths for Sentinel data."""
        if s1_dir:
            self.s1_dir = Path(s1_dir)
        if s2_dir:
            self.s2_dir = Path(s2_dir)
        
        self._validate_setup()
    
    def get_data_paths(self) -> Dict[str, str]:
        """Return data directory paths."""
        return {
            's1_dir': str(self.s1_dir),
            's2_dir': str(self.s2_dir),
            'drive_root': str(self.drive_path)
        }
    
    def list_directory_contents(self, directory: Optional[Path] = None) -> Dict[str, List[str]]:
        """List contents of a directory with file size information."""
        if directory is None:
            directory = self.drive_path
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return {'directories': [], 'files': []}
        
        directories = []
        files = []
        
        for item in directory.iterdir():
            if item.is_dir():
                directories.append(item.name)
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                files.append(f"{item.name} ({size_mb:.1f}MB)")
        
        logger.info(f"Found {len(directories)} directories and {len(files)} files in {directory}")
        return {
            'directories': sorted(directories),
            'files': sorted(files)
        }
    
    def find_sentinel_files(self, satellite_type: str = 'both') -> Dict[str, List[Path]]:
        """
        Find Sentinel files in the data directories.
        
        Args:
            satellite_type: 'S1', 'S2', or 'both'
            
        Returns:
            Dictionary with lists of found files
        """
        found_files = {'S1': [], 'S2': []}
        
        if satellite_type in ['S1', 'both']:
            found_files['S1'] = self._find_files_by_patterns(self.s1_dir, self.s1_patterns)
        
        if satellite_type in ['S2', 'both']:
            found_files['S2'] = self._find_files_by_patterns(self.s2_dir, self.s2_patterns)
        
        # Log results
        for sat_type, files in found_files.items():
            if files:
                logger.info(f"Found {len(files)} {sat_type} files")
                for file in files[:3]:  # Show first 3
                    logger.info(f"  {file.name}")
                if len(files) > 3:
                    logger.info(f"  ... and {len(files) - 3} more")
            else:
                logger.warning(f"No {sat_type} files found")
        
        return found_files
    
    def _find_files_by_patterns(self, directory: Path, patterns: List[str]) -> List[Path]:
        """Find files matching specific patterns."""
        if not directory.exists():
            return []
        
        found_files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in [ext.lower() for ext in self.supported_extensions]:
                filename = file_path.name
                
                # Check if filename matches any pattern
                for pattern in patterns:
                    if re.search(pattern, filename, re.IGNORECASE):
                        found_files.append(file_path)
                        break
        
        return sorted(found_files)
    
    def find_files_by_date(self, date_string: str, satellite_type: str = 'both') -> Dict[str, List[Path]]:
        """
        Find files for a specific date (format: YYYY-MM-DD or YYYYMMDD).
        
        Args:
            date_string: Date in format YYYY-MM-DD or YYYYMMDD
            satellite_type: 'S1', 'S2', or 'both'
        """
        # Normalize date string
        date_clean = date_string.replace('-', '').replace('_', '')
        
        all_files = self.find_sentinel_files(satellite_type)
        date_files = {'S1': [], 'S2': []}
        
        for sat_type, files in all_files.items():
            for file_path in files:
                if date_clean in file_path.name:
                    date_files[sat_type].append(file_path)
        
        logger.info(f"Found {sum(len(files) for files in date_files.values())} files for date {date_string}")
        return date_files
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Union[str, float, int]]:
        """Get detailed information about a file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': f'File does not exist: {file_path}'}
        
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size_mb': stat.st_size / (1024 * 1024),
            'size_bytes': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'extension': file_path.suffix,
            'is_sentinel_1': any(re.search(pattern, file_path.name, re.IGNORECASE) for pattern in self.s1_patterns),
            'is_sentinel_2': any(re.search(pattern, file_path.name, re.IGNORECASE) for pattern in self.s2_patterns)
        }
    
    def validate_file_pairs(self) -> Dict[str, List[Tuple[Path, Path]]]:
        """Find and validate S1/S2 file pairs for the same dates."""
        s1_files = self.find_sentinel_files('S1')['S1']
        s2_files = self.find_sentinel_files('S2')['S2']
        
        # Extract dates from filenames
        s1_dates = {}
        s2_dates = {}
        
        date_pattern = r'(\d{4}-?\d{2}-?\d{2})'
        
        for file in s1_files:
            match = re.search(date_pattern, file.name)
            if match:
                date = match.group(1).replace('-', '')
                if date not in s1_dates:
                    s1_dates[date] = []
                s1_dates[date].append(file)
        
        for file in s2_files:
            match = re.search(date_pattern, file.name)
            if match:
                date = match.group(1).replace('-', '')
                if date not in s2_dates:
                    s2_dates[date] = []
                s2_dates[date].append(file)
        
        # Find matching pairs
        paired_dates = set(s1_dates.keys()) & set(s2_dates.keys())
        single_s1 = set(s1_dates.keys()) - set(s2_dates.keys())
        single_s2 = set(s2_dates.keys()) - set(s1_dates.keys())
        
        pairs = []
        for date in paired_dates:
            # Take first file from each type for simplicity
            pairs.append((s1_dates[date][0], s2_dates[date][0]))
        
        logger.info(f"Found {len(pairs)} S1/S2 pairs, {len(single_s1)} S1-only, {len(single_s2)} S2-only files")
        
        return {
            'pairs': pairs,
            'single_s1_dates': list(single_s1),
            'single_s2_dates': list(single_s2),
            'all_s1_dates': list(s1_dates.keys()),
            'all_s2_dates': list(s2_dates.keys())
        }
    
    def debug_file_structure(self):
        """Debug the entire file structure to help locate data."""
        print("DEBUGGING FILE STRUCTURE")
        print("=" * 50)
        
        # Check drive mount
        if not self._is_drive_mounted():
            print("Google Drive not mounted!")
            return
        
        print("Google Drive mounted")
        
        # Check main directories
        print(f"\nMain Drive Path: {self.drive_path}")
        if self.drive_path.exists():
            contents = self.list_directory_contents(self.drive_path)
            print(f"   Directories: {len(contents['directories'])}")
            print(f"   Files: {len(contents['files'])}")
        else:
            print("   Path does not exist!")
        
        # Check S1 directory
        print(f"\nS1 Directory: {self.s1_dir}")
        if self.s1_dir.exists():
            s1_contents = self.list_directory_contents(self.s1_dir)
            print(f"   Directories: {len(s1_contents['directories'])}")
            print(f"   Files: {len(s1_contents['files'])}")
            
            # Show some files
            if s1_contents['files']:
                print("   Sample files:")
                for file in s1_contents['files'][:3]:
                    print(f"     {file}")
        else:
            print("   Directory does not exist!")
        
        # Check S2 directory
        print(f"\nS2 Directory: {self.s2_dir}")
        if self.s2_dir.exists():
            s2_contents = self.list_directory_contents(self.s2_dir)
            print(f"   Directories: {len(s2_contents['directories'])}")
            print(f"   Files: {len(s2_contents['files'])}")
            
            # Show some files
            if s2_contents['files']:
                print("   Sample files:")
                for file in s2_contents['files'][:3]:
                    print(f"     {file}")
        else:
            print("   Directory does not exist!")
        
        # Search for Sentinel files
        print("\nSEARCHING FOR SENTINEL FILES")
        found_files = self.find_sentinel_files('both')
        
        if found_files['S1']:
            print(f"Found {len(found_files['S1'])} Sentinel-1 files")
        else:
            print("No Sentinel-1 files found")
        
        if found_files['S2']:
            print(f"Found {len(found_files['S2'])} Sentinel-2 files")
        else:
            print("No Sentinel-2 files found")
        
        # Validate pairs
        if found_files['S1'] and found_files['S2']:
            pairs_info = self.validate_file_pairs()
            print(f"\nFound {len(pairs_info['pairs'])} matching S1/S2 pairs")
        
        print("\n" + "=" * 50)
        print("DEBUG COMPLETE")
    
    def extract_file_id(self, drive_url: str) -> Optional[str]:
        """
        Extract file ID from Google Drive URL.
        
        Args:
            drive_url: Google Drive sharing URL
            
        Returns:
            File ID string or None if not found
        """
        # Common Google Drive URL patterns
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/d/([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_url)
            if match:
                return match.group(1)
        
        logger.error(f"Could not extract file ID from URL: {drive_url}")
        return None
    
    def download_from_drive_url(self, drive_url: str, 
                               destination: Optional[Union[str, Path]] = None,
                               filename: Optional[str] = None) -> Optional[Path]:
        """
        Download a file from Google Drive URL to local storage.
        
        Args:
            drive_url: Public Google Drive sharing URL
            destination: Directory to save the file (default: /content/)
            filename: Custom filename (optional)
            
        Returns:
            Path to downloaded file or None if failed
        """
        file_id = self.extract_file_id(drive_url)
        if not file_id:
            return None
        
        # Set destination
        if destination is None:
            destination = Path("/content")
        else:
            destination = Path(destination)
        
        destination.mkdir(parents=True, exist_ok=True)
        
        # Try using gdown first (more reliable for large files)
        try:
            import gdown
            
            if filename:
                output_path = destination / filename
            else:
                output_path = str(destination) + "/"
            
            downloaded_path = gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output=str(output_path),
                quiet=False
            )
            
            if downloaded_path:
                result_path = Path(downloaded_path)
                logger.info(f"Downloaded using gdown: {result_path}")
                return result_path
                
        except ImportError:
            logger.info("gdown not available, trying direct download...")
        except Exception as e:
            logger.warning(f"gdown failed: {e}, trying direct download...")
        
        # Fallback to direct download
        return self._direct_download(file_id, destination, filename)
    
    def _direct_download(self, file_id: str, destination: Path, 
                        filename: Optional[str] = None) -> Optional[Path]:
        """Direct download using requests."""
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            # Handle large file confirmation
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'id': file_id, 'confirm': value}
                    response = session.get(download_url, params=params, stream=True)
                    break
            
            # Determine filename
            if not filename:
                content_disposition = response.headers.get('content-disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                else:
                    filename = f"drive_file_{file_id}.tif"
            
            output_path = destination / filename
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\nDownloading {filename}: {progress:.1f}%", end='', flush=True)
            
            print(f"\nDownloaded: {output_path} ({downloaded_size / (1024*1024):.1f}MB)")
            return output_path
            
        except Exception as e:
            logger.error(f"Direct download failed: {e}")
            return None
    
    def download_s1_files(self, s1_urls: List[str], 
                         destination: Optional[str] = None) -> List[Path]:
        """
        Download multiple Sentinel-1 files from Google Drive URLs.
        
        Args:
            s1_urls: List of Google Drive URLs
            destination: Directory to save files (default: /content/s1_data/)
            
        Returns:
            List of paths to downloaded files
        """
        if destination is None:
            destination = "/content/s1_data"
        
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        for i, url in enumerate(s1_urls):
            logger.info(f"Downloading S1 file {i+1}/{len(s1_urls)}...")
            
            # Generate meaningful filename
            file_id = self.extract_file_id(url)
            filename = f"S1_file_{i+1}_{file_id[:8]}.tif"
            
            downloaded_path = self.download_from_drive_url(url, destination, filename)
            
            if downloaded_path:
                downloaded_files.append(downloaded_path)
                
                # Validate it's a proper TIFF file
                if self._validate_tiff_file(downloaded_path):
                    logger.info(f"Valid TIFF file: {downloaded_path.name}")
                else:
                    logger.warning(f"File may not be a valid TIFF: {downloaded_path.name}")
            else:
                logger.error(f"Failed to download file from: {url}")
        
        logger.info(f"Downloaded {len(downloaded_files)}/{len(s1_urls)} files successfully")
        return downloaded_files
    
    def _validate_tiff_file(self, file_path: Path) -> bool:
        """Validate that a file is a proper TIFF/GeoTIFF."""
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                # Basic validation
                return src.count > 0 and src.width > 0 and src.height > 0
        except ImportError:
            # Fallback: check file size and extension
            return file_path.suffix.lower() in ['.tif', '.tiff'] and file_path.stat().st_size > 1000
        except Exception:
            return False
    
    def setup_user_files(self) -> Dict[str, List[Path]]:
        """
        Download all user's S1 and S2 files in one go.
        
        Returns:
            Dictionary with 'S1' and 'S2' file paths
        """
        # User's S1 URLs
        s1_urls = [
            "https://drive.google.com/file/d/13-EQlzQVhlD7XjJAWHn6iXNnSKufoIjj/view?usp=drive_link",
            "https://drive.google.com/file/d/16_mCAsOp01uMjY22eM7dvAxUkOG0x5D9/view?usp=drive_link"
        ]
        
        # User's S2 URLs
        s2_urls = [
            "https://drive.google.com/file/d/17d7uWnju0FEflk2mncSI5-wrVzXR7MVs/view?usp=drive_link",
            "https://drive.google.com/file/d/1qo2UmGkkuARMS2R-Vvis_8a-WQPzxuqx/view?usp=drive_link",
            "https://drive.google.com/file/d/1ZJAc14zgrS16Smrpa059d6WF5uLkSk8v/view?usp=drive_link",
            "https://drive.google.com/file/d/1ZJjoj-fWD8XamjJGl7WH1NKG8hvJ3EKM/view?usp=drive_link",
            "https://drive.google.com/file/d/17loM0Ndt4P7dK2uH3MCwCjLwKmHxR2KN/view?usp=drive_link",
            "https://drive.google.com/file/d/1sYkTRm12IVzwWgPHdNEoNBzRTp6mjS-7/view?usp=drive_link",
            "https://drive.google.com/file/d/1qKKDtymRcRObZa_IXOyQX-B_Zu8OrrOf/view?usp=drive_link",
            "https://drive.google.com/file/d/1cpmiZNDVpvYxoM6V9gvDtUCqZHhNlP6q/view?usp=drive_link",
            "https://drive.google.com/file/d/1tbLlZs0klgfgHs7CoKEqlQasTvqCAu3L/view?usp=drive_link",
            "https://drive.google.com/file/d/1YiS7kw8yhfjaKCIAxsSTQnR8hOVkaFm4/view?usp=drive_link",
            "https://drive.google.com/file/d/1HxXUSxQGoi6cmtZplx3UDX4CA0eOfx6k/view?usp=drive_link"
        ]
        
        logger.info(f"Downloading {len(s1_urls)} S1 + {len(s2_urls)} S2 files...")
        
        # Download S1 files
        s1_files = self.download_s1_files(s1_urls, "/content/s1_data")
        
        # Download S2 files
        s2_files = self._download_s2_files(s2_urls, "/content/s2_data")
        
        # Update directories
        self.s1_dir = Path("/content/s1_data")
        self.s2_dir = Path("/content/s2_data")
        
        logger.info(f"Downloaded {len(s1_files)} S1 + {len(s2_files)} S2 files")
        
        return {'S1': s1_files, 'S2': s2_files}
    
    def _download_s2_files(self, s2_urls: List[str], destination: str) -> List[Path]:
        """Download S2 files with S2-specific naming."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        for i, url in enumerate(s2_urls):
            file_id = self.extract_file_id(url)
            filename = f"S2_band_{i+1}_{file_id[:8]}.tif"
            
            downloaded_path = self.download_from_drive_url(url, destination, filename)
            if downloaded_path:
                downloaded_files.append(downloaded_path)
        
        return downloaded_files

# Convenience function for quick setup
def setup_drive_loader(auto_mount: bool = True) -> GoogleDriveLoader:
    """
    Quick setup function for GoogleDriveLoader.
    
    Args:
        auto_mount: Whether to automatically mount Google Drive
        
    Returns:
        Configured GoogleDriveLoader instance
    """
    loader = GoogleDriveLoader()
    
    if auto_mount:
        loader.mount_drive()
    
    return loader

# Integration function with data_processing module
def get_data_for_processing(loader: GoogleDriveLoader, date: Optional[str] = None) -> Dict[str, List[Path]]:
    """
    Get data files ready for processing with the data_processing module.
    
    Args:
        loader: GoogleDriveLoader instance
        date: Optional date filter (YYYY-MM-DD format)
        
    Returns:
        Dictionary with S1 and S2 file paths
    """
    if date:
        return loader.find_files_by_date(date)
    else:
        return loader.find_sentinel_files('both')

# Quick setup function for all user files
def setup_all_user_files() -> Tuple[GoogleDriveLoader, Dict[str, List[Path]]]:
    """
    Download all user's S1 and S2 files.
    
    Returns:
        Tuple of (GoogleDriveLoader, {'S1': [paths], 'S2': [paths]})
    """
    # Install gdown if needed
    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "gdown", "-q"])
    
    loader = GoogleDriveLoader()
    files = loader.setup_user_files()
    
    print(f"Ready! {len(files['S1'])} S1 + {len(files['S2'])} S2 files downloaded")
    return loader, files

# Example usage and quick setup for user's specific files
if __name__ == "__main__":
    # Setup all files
    loader, files = setup_all_user_files()
    
    if files['S1'] and files['S2']:
        print(f"SUCCESS! {len(files['S1'])} S1 + {len(files['S2'])} S2 files ready")
        print("Paths: /content/s1_data/ and /content/s2_data/")
    else:
        print("Download failed")
