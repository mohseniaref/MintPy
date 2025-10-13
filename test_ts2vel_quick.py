#!/usr/bin/env python3
"""
Quick test of the timeseries2velocity.py vectorization optimization
"""

import sys
sys.path.insert(0, '/home/mohammad/Software/MintPy/src')

import numpy as np
import time
from scipy import linalg

def test_covariance_vectorization():
    """Test the vectorized covariance calculation"""
    print("🧪 Testing timeseries2velocity.py Covariance Vectorization")
    print("=" * 60)
    
    # Test parameters
    num_date = 50
    num_param = 3
    num_pixel = 5000
    
    print(f"Test: {num_date} dates, {num_param} parameters, {num_pixel:,} pixels")
    
    # Create design matrix and pseudo-inverse
    dates = np.linspace(0, 3, num_date)
    G = np.column_stack([np.ones(num_date), dates, dates**2])
    Gplus = linalg.pinv(G)
    
    # Generate test covariance data
    ts_cov_full = np.random.randn(num_date, num_date, num_pixel).astype(np.float32) * 0.01
    for i in range(num_pixel):
        cov_i = ts_cov_full[:, :, i]
        ts_cov_full[:, :, i] = cov_i @ cov_i.T
    
    # Original method (sample for timing)
    print("\n1. Original pixel-by-pixel method (sampled)...")
    sample_size = min(500, num_pixel)
    start_time = time.time()
    
    m_std_orig = np.zeros((num_param, sample_size), dtype=np.float32)
    for i in range(sample_size):
        ts_covi = ts_cov_full[:, :, i]
        m_cov = np.linalg.multi_dot([Gplus, ts_covi, Gplus.T])
        m_std_orig[:, i] = np.sqrt(np.diag(m_cov))
    
    time_orig = time.time() - start_time
    time_orig_est = time_orig * (num_pixel / sample_size)
    
    print(f"   Sample time: {time_orig:.3f}s ({sample_size:,} pixels)")
    print(f"   Estimated full time: {time_orig_est:.3f}s")
    
    # New vectorized method
    print("\n2. Vectorized batch method...")
    start_time = time.time()
    
    # Vectorized implementation
    ts_cov_batch = ts_cov_full.transpose(2, 0, 1)
    temp = np.matmul(Gplus[np.newaxis, :, :], ts_cov_batch)
    m_cov_batch = np.matmul(temp, Gplus.T[np.newaxis, :, :])
    m_var_valid = np.array([np.diag(m_cov_batch[i]) for i in range(num_pixel)])
    m_std_vec = np.sqrt(m_var_valid.T)
    
    time_vec = time.time() - start_time
    speedup = time_orig_est / time_vec if time_vec > 0 else float('inf')
    
    print(f"   Vectorized time: {time_vec:.3f}s")
    print(f"   🚀 SPEEDUP: {speedup:.1f}x")
    
    # Verify accuracy
    max_diff = np.max(np.abs(m_std_orig - m_std_vec[:, :sample_size]))
    print(f"   Max difference: {max_diff:.2e} (should be ~0)")
    
    if max_diff < 1e-6:
        print("   ✅ Results are numerically identical!")
    else:
        print("   ⚠️  Results show differences")
    
    return speedup

if __name__ == "__main__":
    speedup = test_covariance_vectorization()
    
    print(f"\n{'='*60}")
    print("🎉 timeseries2velocity.py vectorization test completed!")
    print(f"✅ Achieved {speedup:.1f}x speedup for covariance calculation")
    print("="*60)