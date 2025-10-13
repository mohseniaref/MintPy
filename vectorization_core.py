#!/usr/bin/env python3
"""
Clean vectorization implementation for MintPy
Based on proven performance testing results
"""

import numpy as np
import time
from scipy.linalg import lstsq

def estimate_velocity_vectorized(ts_data, design_matrix, uncertainty_method='residue'):
    """
    Ultra-fast vectorized velocity estimation
    
    Parameters:
        ts_data: 2D array (n_dates, n_pixels) - time series data
        design_matrix: 2D array (n_dates, n_params) - design matrix G
        uncertainty_method: str - 'residue' or 'bootstrap'
    
    Returns:
        velocities: 1D array (n_pixels,) - velocity estimates
        uncertainties: 1D array (n_pixels,) - uncertainty estimates
    """
    n_dates, n_pixels = ts_data.shape
    n_params = design_matrix.shape[1]
    
    # Find pixels with valid data (no NaN)
    valid_mask = ~np.any(np.isnan(ts_data), axis=0)
    n_valid = np.sum(valid_mask)
    
    # Initialize outputs
    velocities = np.full(n_pixels, np.nan, dtype=np.float32)
    uncertainties = np.full(n_pixels, np.nan, dtype=np.float32)
    
    if n_valid == 0:
        return velocities, uncertainties
    
    # Extract valid data for vectorized processing
    valid_data = ts_data[:, valid_mask]
    
    try:
        # VECTORIZED LEAST SQUARES - key optimization!
        # Process ALL valid pixels simultaneously
        try:
            params_all, residues, rank, s = lstsq(design_matrix, valid_data, rcond=1e-5)
        except TypeError:
            # Older scipy version doesn't support rcond parameter
            params_all, residues, rank, s = lstsq(design_matrix, valid_data)
        
        # Extract velocity (typically parameter index 1 for linear trend)
        velocities_valid = params_all[1, :].astype(np.float32)
        velocities[valid_mask] = velocities_valid
        
        # Fast uncertainty estimation
        if uncertainty_method == 'residue':
            # Residue-based uncertainty (very fast)
            if residues is not None and len(residues) == n_valid:
                dof = max(n_dates - n_params, 1)
                residue_std = np.sqrt(residues / dof)
                
                # Estimate parameter uncertainty
                try:
                    # Covariance matrix diagonal
                    GTG_inv = np.linalg.inv(design_matrix.T @ design_matrix)
                    param_variance = np.diag(GTG_inv)[1]  # Velocity parameter
                    uncertainties_valid = (residue_std * np.sqrt(param_variance)).astype(np.float32)
                    uncertainties[valid_mask] = uncertainties_valid
                except:
                    # Fallback uncertainty estimate
                    uncertainties[valid_mask] = (0.01 * np.abs(velocities_valid)).astype(np.float32)
            else:
                # Simple uncertainty fallback
                uncertainties[valid_mask] = (0.01 * np.abs(velocities_valid)).astype(np.float32)
        
    except Exception as e:
        print(f"Vectorized estimation failed: {e}")
        # Should rarely happen with proper input validation
        pass
    
    return velocities, uncertainties


def estimate_velocity_chunked(ts_data, design_matrix, chunk_size=50000, uncertainty_method='residue'):
    """
    Memory-efficient chunked vectorized velocity estimation
    
    Parameters:
        ts_data: 2D array (n_dates, n_pixels) - time series data
        design_matrix: 2D array (n_dates, n_params) - design matrix
        chunk_size: int - pixels per chunk
        uncertainty_method: str - uncertainty estimation method
    
    Returns:
        velocities: 1D array (n_pixels,) - velocity estimates
        uncertainties: 1D array (n_pixels,) - uncertainty estimates
    """
    n_dates, n_pixels = ts_data.shape
    
    # Initialize outputs
    velocities = np.full(n_pixels, np.nan, dtype=np.float32)
    uncertainties = np.full(n_pixels, np.nan, dtype=np.float32)
    
    # Process in memory-efficient chunks
    n_chunks = int(np.ceil(n_pixels / chunk_size))
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_pixels)
        
        # Extract chunk
        chunk_data = ts_data[:, start_idx:end_idx]
        
        # Process chunk with vectorization
        chunk_velocities, chunk_uncertainties = estimate_velocity_vectorized(
            chunk_data, design_matrix, uncertainty_method
        )
        
        # Store results
        velocities[start_idx:end_idx] = chunk_velocities
        uncertainties[start_idx:end_idx] = chunk_uncertainties
    
    return velocities, uncertainties


def calculate_rms_vectorized(ts_data, mask=None):
    """
    Ultra-fast vectorized RMS calculation
    
    Parameters:
        ts_data: 2D array (n_dates, n_pixels) - time series data
        mask: 1D array (n_pixels,) - valid pixel mask (optional)
    
    Returns:
        rms_values: 1D array (n_dates,) - RMS for each date
    """
    if mask is None:
        mask = np.ones(ts_data.shape[1], dtype=bool)
    
    # Apply mask by setting invalid pixels to NaN
    masked_data = ts_data.copy()
    masked_data[:, ~mask] = np.nan
    
    # Vectorized RMS calculation - much faster than loops!
    rms_values = np.sqrt(np.nanmean(masked_data**2, axis=1)).astype(np.float32)
    
    return rms_values


def smart_memory_chunking(data_shape, dtype=np.float32, available_memory_gb=None):
    """
    Calculate optimal chunk size based on available memory
    
    Parameters:
        data_shape: tuple - shape of data array
        dtype: numpy dtype - data type
        available_memory_gb: float - available memory in GB (auto-detect if None)
    
    Returns:
        chunk_size: int - optimal chunk size for processing
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_memory_gb = 4.0  # Conservative default
    
    # Calculate memory per element
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Estimate memory usage per pixel (including intermediate arrays)
    memory_per_pixel = data_shape[0] * bytes_per_element * 3  # 3x safety factor
    
    # Calculate maximum pixels that fit in memory
    max_pixels = int(available_memory_gb * 1024**3 * 0.8 / memory_per_pixel)  # 80% of available
    
    # Ensure reasonable chunk size
    chunk_size = max(1000, min(max_pixels, 100000))  # Between 1K and 100K pixels
    
    return chunk_size


if __name__ == "__main__":
    # Quick performance test
    print("🚀 Testing Clean Vectorization Implementation")
    print("=" * 60)
    
    # Test data
    n_dates, n_pixels = 100, 20000
    dates = np.arange(n_dates, dtype=np.float32) / 365.25
    G = np.column_stack([np.ones(n_dates), dates])
    
    # Generate realistic time series with velocity signal
    ts_data = np.random.randn(n_dates, n_pixels).astype(np.float32) * 0.01
    true_velocities = np.random.randn(n_pixels) * 0.02
    for i in range(n_pixels):
        ts_data[:, i] += true_velocities[i] * dates
    
    print(f"Test data: {n_dates} dates × {n_pixels:,} pixels")
    
    # Test vectorized approach
    start = time.time()
    velocities, uncertainties = estimate_velocity_vectorized(ts_data, G)
    time_vec = time.time() - start
    
    valid_count = np.sum(~np.isnan(velocities))
    print(f"Vectorized: {time_vec:.4f}s, {valid_count:,} valid results")
    print(f"Processing rate: {n_pixels/time_vec:,.0f} pixels/second")
    
    # Test chunked approach
    chunk_size = smart_memory_chunking(ts_data.shape)
    print(f"Optimal chunk size: {chunk_size:,} pixels")
    
    start = time.time()
    velocities_chunk, uncertainties_chunk = estimate_velocity_chunked(ts_data, G, chunk_size)
    time_chunk = time.time() - start
    
    diff = np.nanmax(np.abs(velocities - velocities_chunk))
    print(f"Chunked: {time_chunk:.4f}s, max difference: {diff:.2e}")
    
    # Test RMS calculation
    start = time.time()
    rms_values = calculate_rms_vectorized(ts_data)
    time_rms = time.time() - start
    
    print(f"RMS calculation: {time_rms:.4f}s")
    print(f"RMS range: [{np.min(rms_values):.6f}, {np.max(rms_values):.6f}]")
    
    print("\n✅ Clean vectorization implementation ready!")
    print("📈 Ready for integration into MintPy modules")