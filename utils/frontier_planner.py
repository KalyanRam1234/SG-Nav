from queue import PriorityQueue
import numpy as np

class FrontierPathPlanner:
    """
    Plans collision-free paths to frontier targets.
    
    Why separate from occupancy manager:
    - Single Responsibility Principle
    - Reusable with different planners (A*, RRT, etc.)
    """
    
    def __init__(self, occupancy_manager):
        self.occ_mgr = occupancy_manager
    
    def plan_to_frontier(self, start_pos, frontier_pos, max_iterations=5000):
        """
        A* pathfinding to frontier (standard robotics approach).
        
        Why A*:
        - Complete: guarantees finding path if one exists
        - Optimal: finds shortest path
        - Efficient: guided by heuristic to target
        - Industry standard in robotics
        
        Args:
            start_pos: Agent position (world coords)
            frontier_pos: Frontier position (grid coords)
            
        Returns:
            list: Path as world coordinates, or None if unreachable
        """
        start_grid = self.occ_mgr._world_to_grid(start_pos)
        frontier_grid = frontier_pos
        
        open_set = PriorityQueue()
        open_set.put((0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        
        def heuristic(pos):
            return np.sqrt((pos[0] - frontier_grid[0])**2 + (pos[1] - frontier_grid[1])**2)
        
        iteration = 0
        while not open_set.empty() and iteration < max_iterations:
            iteration += 1
            _, current = open_set.get()
            
            if current == frontier_grid:
                # Reconstruct path
                path = [frontier_grid]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return [self.occ_mgr._grid_to_world(p) for p in path]
            
            # Explore 4-neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds and obstacles
                if not self._is_valid_cell(neighbor):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    open_set.put((f_score, neighbor))
        
        return None  # No path found
    
    def _is_valid_cell(self, grid_pos):
        """Check if cell is navigable (free or unknown, not occupied)."""
        grid_x, grid_y = grid_pos
        if not (0 <= grid_x < self.occ_mgr.grid_size and 
                0 <= grid_y < self.occ_mgr.grid_size):
            return False
        return self.occ_mgr.occupancy_grid[grid_y, grid_x] != -1