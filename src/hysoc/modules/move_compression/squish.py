from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import heapq
import math
from hysoc.core.point import Point

@dataclass(order=True)
class PriorityObject:
    priority: float
    index: int = field(compare=False)

class SquishCompressor:
    def __init__(self, capacity: int = 50):
        """
        Initialize the SquishCompressor with a fixed buffer capacity.
        
        Args:
            capacity: maximum number of points to keep in the buffer.
        """
        if capacity < 3:
            raise ValueError("Buffer capacity must be at least 3 to maintain start, end, and one intermediate point.")
        self.capacity = capacity

    def compress(self, points: List[Point], capacity: Optional[int] = None) -> List[Point]:
        """
        Compresses a list of points using the SQUISH algorithm.
        Returns a subset of points that best preserve the trajectory shape within the given capacity.
        
        Args:
            points: List of points to compress.
            capacity: Optional capacity override for this specific compression.
        """
        if not points:
            return []
        
        # Use provided capacity or default to instance capacity
        target_capacity = capacity if capacity is not None else self.capacity
        
        if len(points) <= target_capacity:
            return points

        class Node:
            def __init__(self, point: Point, index: int):
                self.point = point
                self.index = index
                self.prev: Optional['Node'] = None
                self.next: Optional['Node'] = None
                self.priority = float('inf')
                self.removed = False

        nodes: List[Node] = [Node(p, i) for i, p in enumerate(points)]
        
        # Priority queue to efficiently find the point with minimum SED error
        pq: List[PriorityObject] = []
        
        # Maintain a linked buffer of active nodes
        current_buffer_nodes: List[Node] = []
        
        for i in range(len(points)):
            new_node = nodes[i]
            
            if len(current_buffer_nodes) < target_capacity:
                if current_buffer_nodes:
                    last_node = current_buffer_nodes[-1]
                    last_node.next = new_node
                    new_node.prev = last_node
                    
                    if last_node.prev:
                        p = self._compute_priority(last_node.prev.point, last_node.point, new_node.point)
                        last_node.priority = p
                        heapq.heappush(pq, PriorityObject(p, last_node.index))
                        
                current_buffer_nodes.append(new_node)
                
            else:
                # Buffer full. Link new node to end and remove the node with min priority.
                last_node = current_buffer_nodes[-1]
                last_node.next = new_node
                new_node.prev = last_node
                
                if last_node.prev:
                     p = self._compute_priority(last_node.prev.point, last_node.point, new_node.point)
                     last_node.priority = p
                     heapq.heappush(pq, PriorityObject(p, last_node.index))

                current_buffer_nodes.append(new_node)
                
                # Find valid victim (lazy removal from heap)
                while True:
                    victim_item = heapq.heappop(pq)
                    victim_idx = victim_item.index
                    victim_node = nodes[victim_idx]
                    
                    if not victim_node.removed and victim_node.priority == victim_item.priority:
                        break
                
                self._remove_node(victim_node, pq)
                # Note: removing from list is O(K), could be optimized but K is small.
                current_buffer_nodes.remove(victim_node)

        # Collect results by traversing the surviving linked list
        result = []
        head = nodes[0]
        curr = head
        while curr:
            result.append(curr.point)
            curr = curr.next
            
        return result

    def _remove_node(self, node: 'Node', pq: List[PriorityObject]):
        node.removed = True
        prev_node = node.prev
        next_node = node.next
        
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node
            
        if prev_node and prev_node.prev and next_node:
            p = self._compute_priority(prev_node.prev.point, prev_node.point, next_node.point)
            prev_node.priority = p
            heapq.heappush(pq, PriorityObject(p, prev_node.index))
            
        if next_node and next_node.next and prev_node:
            p = self._compute_priority(prev_node.point, next_node.point, next_node.next.point)
            next_node.priority = p
            heapq.heappush(pq, PriorityObject(p, next_node.index))

    def _compute_priority(self, p1: Point, p2: Point, p3: Point) -> float:
        """
        Compute the Synchronized Euclidean Distance (SED) error.
        """
        t1 = p1.timestamp.timestamp()
        t2 = p2.timestamp.timestamp()
        t3 = p3.timestamp.timestamp()
        
        if t1 == t3:
            return 0.0

        ratio = (t2 - t1) / (t3 - t1)
        
        lat_pred = p1.lat + (p3.lat - p1.lat) * ratio
        lon_pred = p1.lon + (p3.lon - p1.lon) * ratio
        
        d_lat = p2.lat - lat_pred
        d_lon = p2.lon - lon_pred
        
        return math.sqrt(d_lat*d_lat + d_lon*d_lon)
