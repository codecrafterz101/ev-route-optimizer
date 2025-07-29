"""
OpenStreetMap Data Collector for Berlin Road Network
"""
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import logging
from typing import Dict, List, Tuple, Optional
import os
import pickle
from datetime import datetime, timedelta
from config import Config
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenStreetMapCollector:
    """Collects and processes OpenStreetMap data for Berlin road network"""
    
    def __init__(self, config: Config):
        self.config = config
        self.berlin_bounds = config.BERLIN_BOUNDS
        self.cache_dir = "data_exports"
        self.cache_file = os.path.join(self.cache_dir, "berlin_road_network_cache.pkl")
        self.cache_ttl_hours = 720  # Cache for 30 days (720 hours) - download once and keep forever
        
        # Configure OSMnx for smaller areas (newer API)
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.max_query_area_size = 50000  # 50km² max area
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_berlin_road_network(self, network_type: str = 'drive', force_refresh: bool = False) -> gpd.GeoDataFrame:
        """
        Extract Berlin road network from OpenStreetMap with caching
        
        Args:
            network_type: Type of network ('drive', 'walk', 'bike')
            force_refresh: Force refresh cache even if valid cache exists
            
        Returns:
            GeoDataFrame with road network data
        """
        logger.info("Getting Berlin road network (with caching)...")
        
        # Check cache first (unless force refresh is requested)
        if not force_refresh:
            cached_data = self._load_cached_road_network()
            if cached_data is not None:
                logger.info("✅ Using cached road network data")
                return cached_data
        
        logger.info("🔄 Cache miss or force refresh - downloading from OpenStreetMap...")
        
        # Define Berlin bounding box
        bbox = [
            self.berlin_bounds['south'],
            self.berlin_bounds['north'],
            self.berlin_bounds['west'],
            self.berlin_bounds['east']
        ]
        
        try:
            # Download road network using OSMnx 2.x API
            # bbox format: (north, south, east, west)
            bbox_tuple = (bbox[1], bbox[0], bbox[3], bbox[2])
            
            logger.info(f"📡 Downloading OSM data for bbox: {bbox_tuple}")
            logger.info(f"📍 Area: ~{abs(bbox[1]-bbox[0])*111:.2f}km x {abs(bbox[3]-bbox[2])*111:.2f}km")
            
            # For very small areas, use a simpler approach
            if abs(bbox[1]-bbox[0]) < 0.01 and abs(bbox[3]-bbox[2]) < 0.01:
                logger.info("🔍 Very small area detected - using simplified download...")
                # Use a point-based approach for tiny areas
                center_lat = (bbox[1] + bbox[0]) / 2
                center_lon = (bbox[3] + bbox[2]) / 2
                G = ox.graph_from_point((center_lat, center_lon), dist=500, network_type=network_type)
            else:
                G = ox.graph_from_bbox(
                    bbox=bbox_tuple,
                    network_type=network_type,
                    simplify=True
                )
            
            logger.info(f"📊 Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
            
            # Convert to GeoDataFrame
            logger.info("🔄 Converting to GeoDataFrame...")
            edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
            logger.info(f"✅ Converted to {len(edges_gdf)} road segments")
            
            # Add elevation data if available
            edges_gdf = self._add_elevation_data(edges_gdf)
            
            # Process road attributes
            logger.info("🔄 Processing road attributes...")
            edges_gdf = self._process_road_attributes(edges_gdf)
            
            # Cache the result
            self._save_cached_road_network(edges_gdf)
            
            logger.info(f"✅ Successfully extracted and cached {len(edges_gdf)} road segments")
            return edges_gdf
            
        except Exception as e:
            logger.error(f"❌ Error extracting road network: {e}")
            raise
    
    def _load_cached_road_network(self) -> Optional[gpd.GeoDataFrame]:
        """Load cached road network if it exists and is valid"""
        if not os.path.exists(self.cache_file):
            logger.info("📁 No cache file found")
            return None
        
        try:
            # Check if cache is still valid
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.cache_file))
            if cache_age > timedelta(hours=self.cache_ttl_hours):
                logger.info(f"⏰ Cache expired (age: {cache_age.total_seconds()/3600:.1f} hours)")
                return None
            
            # Load cached data
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            logger.info(f"📁 Loaded cached road network with {len(cached_data)} segments")
            return cached_data
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading cache: {e}")
            return None
    
    def _save_cached_road_network(self, road_network: gpd.GeoDataFrame):
        """Save road network to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(road_network, f)
            logger.info(f"💾 Cached road network with {len(road_network)} segments")
        except Exception as e:
            logger.warning(f"⚠️ Error saving cache: {e}")
    
    def clear_cache(self):
        """Clear the road network cache"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                logger.info("🗑️ Cache cleared")
        except Exception as e:
            logger.warning(f"⚠️ Error clearing cache: {e}")
    
    def _add_elevation_data(self, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add elevation data to road segments"""
        logger.info("Adding elevation data to road segments...")
        
        try:
            # Get elevation for start and end points of each road segment
            edges_gdf['elevation_start'] = None
            edges_gdf['elevation_end'] = None
            edges_gdf['elevation_gradient'] = None
            
            # Sample a subset for elevation data (due to API limits) - very small for speed
            sample_size = min(10, len(edges_gdf))  # Reduced to just 10 segments
            sample_edges = edges_gdf.sample(n=sample_size, random_state=42)
            
            logger.info(f"📊 Processing elevation data for {sample_size} sample segments...")
            
            for i, (idx, edge) in enumerate(sample_edges.iterrows()):
                if i % 5 == 0:  # Log progress every 5 segments
                    logger.info(f"   Processing elevation segment {i+1}/{sample_size}")
                
                # Get start and end coordinates
                coords = list(edge.geometry.coords)
                start_coord = coords[0]
                end_coord = coords[-1]
                
                # Get elevation (simplified - in practice, use elevation API)
                start_elevation = self._get_elevation_from_coords(start_coord)
                end_elevation = self._get_elevation_from_coords(end_coord)
                
                # Calculate gradient
                distance = edge.geometry.length
                if distance > 0:
                    gradient = (end_elevation - start_elevation) / distance
                else:
                    gradient = 0
                
                edges_gdf.loc[idx, 'elevation_start'] = start_elevation
                edges_gdf.loc[idx, 'elevation_end'] = end_elevation
                edges_gdf.loc[idx, 'elevation_gradient'] = gradient
            
            logger.info("✅ Elevation data processing completed")
            return edges_gdf
            
        except Exception as e:
            logger.warning(f"Could not add elevation data: {e}")
            return edges_gdf
    
    def _get_elevation_from_coords(self, coords: Tuple[float, float]) -> float:
        """Get elevation for coordinates (placeholder implementation)"""
        # In practice, use elevation API like Open-Elevation or Google Elevation API
        # For now, return a placeholder value
        return 50.0  # meters above sea level
    
    def _process_road_attributes(self, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process and standardize road attributes"""
        logger.info("Processing road attributes...")
        
        # Standardize speed limits
        edges_gdf['speed_limit_kmh'] = edges_gdf['maxspeed'].apply(self._parse_speed_limit)
        
        # Add road type classification
        edges_gdf['road_type'] = edges_gdf['highway'].apply(self._classify_road_type)
        
        # Add energy efficiency factors
        edges_gdf['energy_efficiency_factor'] = edges_gdf.apply(self._calculate_energy_efficiency, axis=1)
        
        # Add traffic capacity
        edges_gdf['traffic_capacity'] = edges_gdf.apply(self._estimate_traffic_capacity, axis=1)
        
        return edges_gdf
    
    def _parse_speed_limit(self, speed_str: str) -> float:
        """Parse speed limit string to float"""
        if pd.isna(speed_str):
            return 50.0  # Default Berlin speed limit
        
        try:
            # Handle various speed limit formats
            if isinstance(speed_str, str):
                if 'mph' in speed_str.lower():
                    return float(speed_str.replace('mph', '').strip()) * 1.60934
                elif 'kmh' in speed_str.lower() or 'km/h' in speed_str.lower():
                    return float(speed_str.replace('kmh', '').replace('km/h', '').strip())
                else:
                    return float(speed_str)
            else:
                return float(speed_str)
        except:
            return 50.0  # Default fallback
    
    def _classify_road_type(self, highway_type: str) -> str:
        """Classify road type for energy consumption modeling"""
        if pd.isna(highway_type):
            return 'residential'
        
        highway_type = str(highway_type).lower()
        
        if highway_type in ['motorway', 'trunk']:
            return 'highway'
        elif highway_type in ['primary', 'secondary']:
            return 'arterial'
        elif highway_type in ['tertiary', 'residential']:
            return 'local'
        else:
            return 'other'
    
    def _calculate_energy_efficiency(self, row) -> float:
        """Calculate energy efficiency factor for road segment"""
        base_efficiency = 1.0
        
        # Road type impact
        road_type = row.get('road_type', 'local')
        if road_type == 'highway':
            base_efficiency *= 0.9  # Highways are more efficient
        elif road_type == 'arterial':
            base_efficiency *= 0.95
        elif road_type == 'local':
            base_efficiency *= 1.1  # Local roads less efficient due to stops
        
        # Speed limit impact
        speed_limit = row.get('speed_limit_kmh', 50)
        if speed_limit > 80:
            base_efficiency *= 0.85  # High speeds less efficient
        elif speed_limit < 30:
            base_efficiency *= 1.2  # Very low speeds less efficient
        
        # Elevation gradient impact
        gradient = row.get('elevation_gradient', 0)
        if gradient and gradient > 0.05:  # 5% uphill
            base_efficiency *= 1.3
        elif gradient and gradient < -0.05:  # 5% downhill
            base_efficiency *= 0.8
        
        return base_efficiency
    
    def _estimate_traffic_capacity(self, row) -> str:
        """Estimate traffic capacity based on road attributes"""
        road_type = row.get('road_type', 'local')
        lanes_raw = row.get('lanes', 1)
        
        # Convert lanes to integer with error handling
        try:
            if pd.isna(lanes_raw):
                lanes = 1
            elif isinstance(lanes_raw, str):
                # Handle string values like "2", "3;4", "2 lanes", etc.
                lanes_str = lanes_raw.split(';')[0].split()[0]  # Take first number
                lanes = int(float(lanes_str))  # Convert via float to handle "2.0"
            else:
                lanes = int(lanes_raw)
        except (ValueError, TypeError):
            lanes = 1  # Default fallback
        
        if road_type == 'highway':
            return 'high'
        elif road_type == 'arterial' and lanes >= 2:
            return 'medium'
        else:
            return 'low'
    
    def get_charging_stations(self, force_refresh: bool = False) -> gpd.GeoDataFrame:
        """Extract charging station locations from OpenStreetMap with caching"""
        logger.info("Getting charging stations (with caching)...")
        
        # Check cache first (unless force refresh is requested)
        if not force_refresh:
            cached_stations = self._load_cached_charging_stations()
            if cached_stations is not None:
                logger.info("✅ Using cached charging stations data")
                return cached_stations
        
        logger.info("🔄 Cache miss or force refresh - downloading from OpenStreetMap...")
        
        try:
            # FAST APPROACH: Use multiple points instead of large bounding box
            # This is much faster than bbox queries
            logger.info("🚀 Using fast point-based queries...")
            
            # Define multiple points across Berlin for better coverage
            berlin_points = [
                (52.52, 13.41),   # Central Berlin
                (52.54, 13.38),   # North Berlin
                (52.50, 13.44),   # South Berlin
                (52.52, 13.35),   # West Berlin
                (52.52, 13.47),   # East Berlin
            ]
            
            all_stations = []
            
            for i, point in enumerate(berlin_points):
                logger.info(f"📡 Downloading point {i+1}/{len(berlin_points)}: {point}")
                
                try:
                    # Use point-based query (much faster than bbox)
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("OSM query timed out")
                    
                    # Set 30 second timeout per point
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)
                    
                    stations = ox.features_from_point(
                        point,
                        tags={'amenity': 'charging_station'},
                        dist=300  # 300m radius per point (balanced speed/coverage)
                    )
                    
                    signal.alarm(0)  # Cancel timeout
                    
                    if len(stations) > 0:
                        all_stations.append(stations)
                        logger.info(f"✅ Found {len(stations)} stations at point {i+1}")
                    
                except (TimeoutError, Exception) as e:
                    signal.alarm(0)  # Cancel timeout if it was set
                    logger.warning(f"⚠️ Error at point {i+1}: {e}")
                    continue
            
            # Combine all results
            if all_stations:
                import pandas as pd
                charging_stations = pd.concat(all_stations, ignore_index=True)
                charging_stations = gpd.GeoDataFrame(charging_stations)
                
                # Remove duplicates based on geometry
                charging_stations = charging_stations.drop_duplicates(subset=['geometry'])
                
                logger.info(f"✅ Combined {len(charging_stations)} unique charging stations")
            else:
                logger.warning("No charging stations found in OpenStreetMap")
                return gpd.GeoDataFrame()
            
            # Process charging station data
            charging_stations = self._process_charging_stations(charging_stations)
            
            # Cache the result
            self._save_cached_charging_stations(charging_stations)
            
            logger.info(f"✅ Found and cached {len(charging_stations)} charging stations")
            return charging_stations
            
        except Exception as e:
            logger.error(f"❌ Error extracting charging stations: {e}")
            return gpd.GeoDataFrame()
    
    def _load_cached_charging_stations(self) -> Optional[gpd.GeoDataFrame]:
        """Load cached charging stations if they exist and are valid"""
        cache_file = os.path.join(self.cache_dir, "berlin_charging_stations_cache.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Check if cache is still valid
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age > timedelta(hours=self.cache_ttl_hours):
                return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            logger.info(f"📁 Loaded cached charging stations with {len(cached_data)} stations")
            return cached_data
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading charging stations cache: {e}")
            return None
    
    def _save_cached_charging_stations(self, charging_stations: gpd.GeoDataFrame):
        """Save charging stations to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, "berlin_charging_stations_cache.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(charging_stations, f)
            logger.info(f"💾 Cached charging stations with {len(charging_stations)} stations")
        except Exception as e:
            logger.warning(f"⚠️ Error saving charging stations cache: {e}")
    
    def _process_charging_stations(self, stations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process charging station attributes"""
        # Add charging power information
        stations_gdf['charging_power_kw'] = stations_gdf['socket:type2:output'].apply(
            self._parse_charging_power
        )
        
        # Add charging type
        stations_gdf['charging_type'] = stations_gdf['charging_power_kw'].apply(
            self._classify_charging_type
        )
        
        # Add availability status
        stations_gdf['is_available'] = True  # Default assumption
        
        # Add missing fields that data manager expects
        stations_gdf['station_id'] = [f'osm_station_{i}' for i in range(len(stations_gdf))]
        stations_gdf['price_per_kwh'] = np.random.uniform(0.25, 0.45, len(stations_gdf))
        stations_gdf['operator'] = stations_gdf.get('operator', 'Unknown').fillna('Unknown')
        stations_gdf['connector_types'] = stations_gdf['charging_type'].apply(
            lambda x: ['Type 2', 'CCS'] if x == 'fast' else ['Type 2']
        )
        
        return stations_gdf
    
    def _parse_charging_power(self, power_str: str) -> float:
        """Parse charging power string to float"""
        if pd.isna(power_str):
            return 22.0  # Default to standard charging
        
        try:
            return float(power_str.replace('kW', '').strip())
        except:
            return 22.0
    
    def _classify_charging_type(self, power: float) -> str:
        """Classify charging type based on power"""
        if power >= 50:
            return 'fast'
        elif power >= 11:
            return 'standard'
        else:
            return 'slow' 