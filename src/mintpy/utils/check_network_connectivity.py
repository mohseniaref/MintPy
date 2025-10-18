#!/usr/bin/env python3
"""
Check network connectivity and find disconnected components in the interferogram network.
"""

import h5py
import numpy as np
from collections import defaultdict
import sys

def find_connected_components(graph, all_dates):
    """Find all connected components using DFS."""
    visited = set()
    components = []
    
    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for date in all_dates:
        if date not in visited:
            component = set()
            dfs(date, component)
            components.append(sorted(component))
    
    return components

def main():
    # Read the ifgramStack
    print('Reading ifgramStack.h5...')
    with h5py.File('inputs/ifgramStack.h5', 'r') as f:
        # Read date12 list - each element is like "20170115_20170127"
        date12_array = f['date'][()]
        # Handle both string and byte arrays - check if it's 2D array
        if date12_array.ndim == 2:
            # It's stored as pair of dates
            date12_list = ['_'.join([d.decode('utf-8') if isinstance(d, bytes) else str(d) 
                                     for d in pair]) for pair in date12_array]
        elif isinstance(date12_array[0], bytes):
            date12_list = [d.decode('utf-8') for d in date12_array]
        else:
            date12_list = [str(d) for d in date12_array]
        
        drop_ifgram = f['dropIfgram'][()]
        
    # Get kept interferograms only
    date12_kept = [d for d, drop in zip(date12_list, drop_ifgram) if not drop]
    print(f'Total interferograms: {len(date12_list)}')
    print(f'Kept interferograms: {len(date12_kept)}')
    print(f'Dropped interferograms: {len(date12_list) - len(date12_kept)}')

    # Get all unique dates
    dates_all = sorted(list(set([d for date12 in date12_kept for d in date12.split('_')])))
    print(f'\nTotal unique dates: {len(dates_all)}')
    print(f'Date range: {dates_all[0]} to {dates_all[-1]}')

    # Build adjacency graph
    print('\n' + '='*70)
    print('CHECKING NETWORK CONNECTIVITY')
    print('='*70)
    
    graph = defaultdict(set)
    for date12 in date12_kept:
        date1, date2 = date12.split('_')
        graph[date1].add(date2)
        graph[date2].add(date1)

    # Find connected components
    components = find_connected_components(graph, dates_all)

    print(f'\nNumber of connected components: {len(components)}')
    print()

    for i, comp in enumerate(components, 1):
        print(f'Component {i}: {len(comp)} dates')
        print(f'  Date range: {comp[0]} to {comp[-1]}')
        if len(comp) <= 10:
            print(f'  Dates: {comp}')
        else:
            print(f'  First 5 dates: {comp[:5]}')
            print(f'  Last 5 dates: {comp[-5:]}')
        
        # Count interferograms in this component
        comp_set = set(comp)
        ifgs_in_comp = [d for d in date12_kept if d.split('_')[0] in comp_set and d.split('_')[1] in comp_set]
        print(f'  Interferograms in component: {len(ifgs_in_comp)}')
        print()

    # Find dates with minimal connections (potential weak points)
    print('='*70)
    print('DATES WITH FEW CONNECTIONS (potential weak points):')
    print('='*70)
    connection_counts = [(date, len(graph[date])) for date in dates_all]
    connection_counts.sort(key=lambda x: x[1])

    print(f'{"Date":<12} {"Connections":<15} {"Connected to"}')
    print('-'*70)
    for date, count in connection_counts[:20]:  # Show 20 weakest
        neighbors = sorted(list(graph[date]))[:5]
        neighbors_str = ', '.join(neighbors)
        if len(graph[date]) > 5:
            neighbors_str += ', ...'
        print(f'{date:<12} {count:<15} {neighbors_str}')

    # Identify bridge dates (if removed, would disconnect network)
    if len(components) > 1:
        print('\n' + '='*70)
        print('NETWORK IS DISCONNECTED!')
        print('='*70)
        print(f'The network has {len(components)} separate components.')
        print('These groups of dates are not connected to each other.')
        
        # Find potential bridges between components
        print('\nTo connect all components, you would need interferograms between:')
        for i in range(len(components)-1):
            for j in range(i+1, len(components)):
                comp1_dates = components[i]
                comp2_dates = components[j]
                print(f'\n  Component {i+1} ↔ Component {j+1}:')
                print(f'    Dates in Component {i+1}: {comp1_dates[0]} to {comp1_dates[-1]}')
                print(f'    Dates in Component {j+1}: {comp2_dates[0]} to {comp2_dates[-1]}')
                
                # Suggest closest dates
                from datetime import datetime
                def date_to_datetime(d):
                    return datetime.strptime(d, '%Y%m%d')
                
                # Find closest pair between components
                min_gap = float('inf')
                closest_pair = None
                for d1 in comp1_dates:
                    for d2 in comp2_dates:
                        dt1 = date_to_datetime(d1)
                        dt2 = date_to_datetime(d2)
                        gap = abs((dt2 - dt1).days)
                        if gap < min_gap:
                            min_gap = gap
                            closest_pair = (d1, d2)
                
                if closest_pair:
                    print(f'    Suggested connection: {closest_pair[0]}_{closest_pair[1]} ({min_gap} days)')
    else:
        print('\n' + '='*70)
        print('✓ NETWORK IS FULLY CONNECTED!')
        print('='*70)
        print('All dates are connected through at least one path.')

    # Save connection statistics to file
    print('\n' + '='*70)
    print('Saving connection statistics to network_connectivity.txt...')
    with open('network_connectivity.txt', 'w') as f:
        f.write('Network Connectivity Analysis\n')
        f.write('='*70 + '\n\n')
        f.write(f'Total interferograms: {len(date12_list)}\n')
        f.write(f'Kept interferograms: {len(date12_kept)}\n')
        f.write(f'Total unique dates: {len(dates_all)}\n')
        f.write(f'Date range: {dates_all[0]} to {dates_all[-1]}\n')
        f.write(f'Number of connected components: {len(components)}\n\n')
        
        for i, comp in enumerate(components, 1):
            f.write(f'Component {i}: {len(comp)} dates\n')
            f.write(f'  Date range: {comp[0]} to {comp[-1]}\n')
            f.write(f'  All dates: {", ".join(comp)}\n\n')
        
        f.write('\nConnection counts per date:\n')
        f.write(f'{"Date":<12} {"Connections":<15}\n')
        f.write('-'*30 + '\n')
        for date, count in sorted(connection_counts):
            f.write(f'{date:<12} {count:<15}\n')
    
    print('✓ Saved to network_connectivity.txt')

if __name__ == '__main__':
    main()
