# **Implementational Architecture for Kinetic Ceiling Reconstruction of Scanned Buildings using CGAL and LLM-Assisted Environments**

Reconstructing the structural architecture of building interiors—specifically the ceiling plane—from raw three-dimensional point clouds represents an enduring challenge in computer graphics, photogrammetry, and building information modeling (BIM).1 The acquisition of interior scans is frequently plagued by severe sensor occlusions.1 Terrestrial laser scanners (LiDAR) and depth cameras operating from floor-level or tripod positions face significant line-of-sight obstructions caused by hanging light fixtures, HVAC ductwork, structural beams, columns, and furniture.1 Consequently, point clouds representing ceilings are highly fragmented, exhibiting large spatial gaps, heterogeneous point densities, and substantial noise.1  
The Kinetic Shape Reconstruction framework, developed by Jean-Philippe Bauchet and Florent Lafarge, provides a robust, watertight, and piece-wise planar solution to these structural deficiencies.1 Rather than executing an exhaustive binary space partition by intersecting the infinite support planes of all detected shapes across the entire bounding volume—which incurs cubic computational complexity and generates thousands of redundant, tiny cells—the kinetic approach models planar shapes as active polygons.1 These polygons expand over time within their respective support planes until they collide, forming localized, meaningful polyhedral subdivisions.1  
This document provides a comprehensive technical guide for implementing this framework. It details the underlying mathematical mechanics, spatial partitioning strategies, and integration within an AI-assisted development environment like Cursor, utilizing the Computational Geometry Algorithms Library (CGAL).5

## **Theoretical Foundation of Kinetic Shape Reconstruction**

The core innovation of the kinetic shape reconstruction algorithm lies in its departure from exhaustive slicing methods. Traditional slicing algorithms partition the bounding box of a point cloud by intersecting the supporting planes of all detected shapes. For a configuration of ![][image1] planar shapes, the exact geometric computation of such a 3D arrangement has a theoretical complexity of ![][image2].7 When processing complex architectural scenes containing hundreds of distinct planar regions, this cubic growth quickly exhausts system memory and leads to severe computational bottlenecks.1 Moreover, the resulting partition contains many tiny, highly anisotropic cells, which complicates the subsequent surface extraction process.

### **The Kinetic Space Partition Paradigm**

The kinetic space partition resolves these performance bottlenecks by initiating a simulation where detected planar shapes, bounded by their initial 2D convex hulls, expand outward at a constant velocity within their supporting planes.1 This kinetic approach restricts plane extensions to local regions, avoiding the global partitioning overhead of exhaustive methods.5

       Initial State                   Kinetic Growth                  Final Collision  
    \+-----------------+              \+-----------------+              \+-----------------+  
    |                 |              |   \======\>       |              |=================|  
    |   Polygon A     |              |   Polygon A     |              |    Polygon A    |  
    |   \-------       |     \==\>      |   \---------     |     \==\>      |-----------------|  
    |                 |              |                 |              |                 |  
    |       \-------   |              |     \---------   |              |=================|  
    |       Polygon B |              |     Polygon B   |              |    Polygon B    |  
    \+-----------------+              \+-----------------+              \+-----------------+

This localized growth is governed by a kinetic data structure where geometric primitives are parameterized as continuous functions of time, ![][image3]. For each active primitive ![][image4], a certificate function ![][image5] is defined to assert that the primitive remains free of invalid intersections with any other active primitive ![][image6] :  
![][image7]  
The predicate function ![][image8] yields a value of ![][image9] at the exact temporal instance of a collision—defined as when the minimum Euclidean distance between the boundary of primitive ![][image4] and the closed boundary of primitive ![][image6] reaches zero—and ![][image10] otherwise.  
To track these occurrences without performing expensive continuous 3D collision queries, the formulation projects the 3D collision problem into ![][image1] collaborative 2D line-segment propagation problems. Because two non-coplanar 3D polygons can only collide along the 3D line representing the intersection of their infinite supporting planes, the 3D collision is calculated as a 2D point-to-line distance within the local plane of each polygon. This algorithmic reformulation dramatically accelerates priority queue updates during the simulation.

### **Collision Typology and Primitive Updates**

When a collision event is popped from the priority queue, the topological state of the partition is updated according to four primary geometric transitions :

* **Type A (Vertex-to-Polygon Collision):** A vertex of the growing source polygon collides with the target polygon. The vertex is replaced by two sliding vertices constrained to propagate along the line of intersection.  
* **Type B (Sliding Vertex-to-Polygon Collision):** A sliding vertex collides with another target plane. Its propagation direction is redirected to follow the new intersection line, and a static "frozen" vertex is instantiated at the intersection junction.  
* **Type C (Sliding-to-Sliding Vertex Collision):** Two sliding vertices meet along an intersection line. A frozen vertex is instantiated at their coordinates, permanently halting expansion along that local sector of the polygon.  
* **Type D (Edge-to-Edge Collision):** The edge of a source polygon collides with the edge of a target polygon. The polygons are split along their line of intersection, resulting in the creation of new sliding and frozen vertices for the split sub-polygons.

To compare these fundamental approaches, the performance and structural characteristics of exhaustive slicing are contrasted with kinetic partitioning in the table below:

| Evaluation Metric | Exhaustive Plane Slicing | Kinetic Space Partitioning (K=2) |
| :---- | :---- | :---- |
| **Spatial Complexity** | **![][image2]** plane arrangement | Localized sub-cubic cell complex |
| **Partitional Cell Count** | High (tens of thousands of micro-cells) | Low (order of magnitude fewer, isotropic cells) |
| **Memory Peak (150+ Planes)** | **![][image11]** (frequently crashes) | ![][image12] |
| **Boundary Optimization** | Prone to failure with unbounded planes | Bounded by initial hulls and local growth |
| **Missing Data Tolerance** | Poor (leaves gaps or generates artifacts) | High (bridges gaps via bounded propagation) |

## **Structural Constraints and Data Imperfections in Ceiling Scans**

Implementing a robust reconstruction pipeline specifically for building ceilings requires addressing several domain-specific geometric constraints.1 Building ceilings are typically characterized by large horizontal planes that must connect seamlessly with vertical wall structures.1

### **The Inversion Challenge of Interior Scans**

A key difficulty in interior building reconstruction is the physical orientation of the scan data.8 In standard object reconstruction (such as industrial parts or CAD models), the point cloud represents a closed, solid exterior shell.1 The space surrounding the bounding box is empty, and the interior of the shell is solid.1  
For interior building scans, this relationship is inverted.8 The scanner is positioned *inside* the rooms, meaning the interior volume is empty space, while the region beyond the walls and above the ceiling is solid structure.8 Consequently, point normals estimated from interior scans typically point inward toward the scanner.  
This inversion requires careful configuration of the boundary conditions during the surface extraction phase.8 If the empty interior space is allowed to "leak" through unobserved boundary faces of the bounding box, the graph-cut solver may fail to reconstruct the ceiling plane entirely, or it may extend the room volume infinitely upward.8 Presetting the boundary face above the ceiling (ZMAX) to inside (occupied) acts as a structural boundary, forcing the final surface to align precisely with the detected ceiling plane.8

### **Geometric Regularities and Tandem Plane Optimization**

Because ceilings and walls are subject to strict architectural design rules, raw plane detections must be regularized prior to partitioning.5 The ceiling plane must be perpendicular to the surrounding walls and parallel to the floor.4  
To enforce these constraints, a shape regularization phase is executed immediately after shape detection.5 This step aligns the detected planes to dominant directions (typically aligned with a Manhattan World frame).4 It forces parallel planes that lie within a small offset distance to become coplanar, and adjusts nearly perpendicular wall-ceiling junctions to be exactly ![][image13].1 This regularization simplifies the downstream kinetic simulation by reducing the number of redundant, close-parallel planes that would otherwise trigger high-frequency, complex collision events.1

## **Architectural Integration within LLM-Assisted Development Environments**

Integrating the CGAL Kinetic Surface Reconstruction package within modern LLM-assisted development environments like Cursor requires a structured approach to project configuration, dependency management, and type selection.5 Cursor's code generation capabilities are highly effective when provided with clear specifications for CGAL's exact computation kernels and memory management patterns.1

### **Managing C++ Geometry Dependencies with vcpkg in Cursor**

The pipeline relies on several external libraries for exact arithmetic and graph-cut optimization.1 To set up the workspace, the C++ package manager vcpkg is used to acquire CGAL, Boost, GMP, and MPFR.6 A typical vcpkg.json manifest file for this setup is configured as follows:

JSON  
{  
  "name": "kinetic-ceiling-reconstruction",  
  "version": "1.0.0",  
  "dependencies": \[  
    "boost",  
    "gmp",  
    "mpfr",  
    "cgal"  
  \]  
}

When writing code in Cursor, specifying this manifest allows the IDE to correctly locate headers and provide accurate auto-completion for CGAL's templates.6

### **Exact vs. Inexact Kernels in Kinetic Simulations**

A critical design choice in CGAL is the selection of geometric kernels.12 For the input point cloud and initial normal estimation, an inexact constructions kernel (CGAL::Exact\_predicates\_inexact\_constructions\_kernel) is preferred to minimize computational overhead.13  
However, the kinetic space partitioning simulation involves continuous updates to polygon boundaries, vertex splits, and collision event scheduling.1 These operations are highly sensitive to floating-point drift. Even minor numerical inaccuracies can lead to self-intersecting polygons or infinite loops in the priority queue, causing the simulation to crash.  
To guarantee numerical stability, the kinetic space partitioner must utilize an exact constructions kernel (CGAL::Exact\_predicates\_exact\_constructions\_kernel).7 This kernel represents coordinates as exact rational numbers, ensuring that all collision calculations and topological updates are mathematically rigorous and free of floating-point errors.1

## **Complete Implementational Blueprint**

The following section presents a complete implementational blueprint for the ceiling reconstruction pipeline.14 This setup is designed for direct compilation and deployment within a C++17 environment configured in Cursor.6

### **Build System Configuration**

The build system is managed via CMakeLists.txt. It detects the vcpkg dependencies and configures the appropriate compilation flags for exact arithmetic.

CMake  
cmake\_minimum\_required(VERSION 3.16)  
project(KineticCeilingReconstruction)

set(CMAKE\_CXX\_STANDARD 17)  
set(CMAKE\_CXX\_STANDARD\_REQUIRED ON)

\# Find CGAL and its required dependencies  
find\_package(CGAL REQUIRED COMPONENTS Core)

\# Define the target executable  
add\_executable(reconstruct\_ceiling main.cpp)

\# Link CGAL and target libraries  
target\_link\_libraries(reconstruct\_ceiling PRIVATE CGAL::CGAL CGAL::CGAL\_Core)

### **Pipeline Source Implementation**

The complete C++ implementation file (main.cpp) coordinates the reconstruction process.14 It loads the raw point cloud, estimates and orients normals, detects and regularizes planar shapes, executes the kinetic simulation, and extracts the watertight ceiling surface using a graph-cut solver.5

C++  
\#**include** \<CGAL/Exact\_predicates\_inexact\_constructions\_kernel.h\>  
\#**include** \<CGAL/Exact\_predicates\_exact\_constructions\_kernel.h\>  
\#**include** \<CGAL/Kinetic\_surface\_reconstruction\_3.h\>  
\#**include** \<CGAL/Point\_set\_3.h\>  
\#**include** \<CGAL/Point\_set\_3/IO.h\>  
\#**include** \<CGAL/pca\_estimate\_normals.h\>  
\#**include** \<CGAL/mst\_orient\_normals.h\>  
\#**include** \<CGAL/bounding\_box.h\>  
\#**include** \<CGAL/IO/polygon\_soup\_io.h\>  
\#**include** \<iostream\>  
\#**include** \<vector\>  
\#**include** \<map\>  
\#**include** \<string\>

// Inexact kernel for efficient preprocessing of the input point cloud  
using Geom\_kernel \= CGAL::Exact\_predicates\_inexact\_constructions\_kernel;  
// Exact kernel to ensure numerical stability during kinetic partitioning  
using Intersection\_kernel \= CGAL::Exact\_predicates\_exact\_constructions\_kernel;

using FT \= typename Geom\_kernel::FT;  
using Point\_3 \= typename Geom\_kernel::Point\_3;  
using Vector\_3 \= typename Geom\_kernel::Vector\_3;  
using Point\_set \= CGAL::Point\_set\_3\<Point\_3\>;

using Point\_map \= typename Point\_set::Point\_map;  
using Normal\_map \= typename Point\_set::Vector\_map;

// Define the Kinetic Surface Reconstruction pipeline class  
using KSR \= CGAL::Kinetic\_surface\_reconstruction\_3\<  
    Geom\_kernel,   
    Point\_set,   
    Point\_map,   
    Normal\_map,   
    Intersection\_kernel  
\>;

int main(int argc, char\*\* argv) {  
    if (argc \< 3) {  
        std::cerr \<\< "Usage: " \<\< argv \<\< " \<input\_ply\_file\> \<output\_mesh\_file\>" \<\< std::endl;  
        return EXIT\_FAILURE;  
    }

    const std::string input\_path \= argv;  
    const std::string output\_path \= argv;

    // 1\. Load the raw point cloud data  
    Point\_set point\_set;  
    if (\!CGAL::IO::read\_point\_set(input\_path, point\_set)) {  
        std::cerr \<\< "Error: Failed to parse the input file: " \<\< input\_path \<\< std::endl;  
        return EXIT\_FAILURE;  
    }  
    std::cout \<\< "Successfully loaded " \<\< point\_set.size() \<\< " points." \<\< std::endl;

    // 2\. Compute and orient point normals if missing from the scan  
    if (\!point\_set.has\_normal\_map()) {  
        std::cout \<\< "Normals missing. Estimating local normals via PCA..." \<\< std::endl;  
        point\_set.add\_normal\_map();  
        // Estimate normals using the 12 nearest neighbors  
        CGAL::pca\_estimate\_normals\<CGAL::Parallel\_if\_available\_tag\>(point\_set, 12);  
          
        std::cout \<\< "Orienting normals using a Minimum Spanning Tree..." \<\< std::endl;  
        CGAL::mst\_orient\_normals(point\_set, 12);  
    }

    // 3\. Compute the bounding box diagonal to scale tolerances dynamically  
    CGAL::Bbox\_3 bbox \= CGAL::bbox\_3(  
        CGAL::make\_transform\_iterator\_from\_property\_map(point\_set.begin(), point\_set.point\_map()),  
        CGAL::make\_transform\_iterator\_from\_property\_map(point\_set.end(), point\_set.point\_map())  
    );

    const FT diag \= CGAL::approximate\_sqrt(  
        (bbox.xmax() \- bbox.xmin()) \* (bbox.xmax() \- bbox.xmin()) \+  
        (bbox.ymax() \- bbox.ymin()) \* (bbox.ymax() \- bbox.ymin()) \+  
        (bbox.zmax() \- bbox.zmin()) \* (bbox.zmax() \- bbox.zmin())  
    );  
    std::cout \<\< "Bounding box diagonal: " \<\< diag \<\< " meters." \<\< std::endl;

    // 4\. Configure pipeline parameters for building interior scans  
    auto params \= CGAL::parameters::maximum\_distance(0.01 \* diag) // Distance tolerance for plane detection (1% of diagonal)  
       .maximum\_angle(15.0)                                     // Normal deviation angle in degrees  
       .minimum\_region\_size(static\_cast\<std::size\_t\>(point\_set.size() \* 0.01)) // Reject noisy regions (1% of points)  
       .regularize\_parallelism(true)                            // Enforce parallel walls and floors  
       .regularize\_coplanarity(true)                            // Merge parallel coplanar segments  
       .regularize\_orthogonality(true)                          // Enforce wall-to-ceiling perpendicularity  
       .reorient\_bbox(true)                                     // Align bounding box with dominant wall directions  
       .max\_octree\_depth(3)                                     // Enable octree subdivision to speed up partitioning  
       .max\_octree\_node\_size(40);                               // Subdivision threshold

    // 5\. Initialize the Kinetic Surface Reconstruction pipeline  
    KSR ksr(point\_set, params);

    // 6\. Detect planar shapes and execute kinetic space partitioning  
    std::cout \<\< "Executing shape detection and regularization..." \<\< std::endl;  
    // Parameter k \= 2 allows ceiling polygons to propagate past minor structural gaps  
    const std::size\_t k\_intersections \= 2;  
    ksr.detection\_and\_partition(k\_intersections, params);  
    std::cout \<\< "Detected " \<\< ksr.detected\_planar\_shapes().size() \<\< " planar shapes." \<\< std::endl;

    // 7\. Execute Graph-Cut labeling with boundary conditions  
    // Lambda parameter (0.75) trades data faithfulness for lower mesh complexity (flatter surfaces)  
    const FT lambda \= 0.75;  
      
    // Define the boundary conditions to handle inverted interior scans  
    std::map\<typename KSR::KSP::Face\_support, bool\> external\_nodes;  
      
    // Preset ZMAX (above the ceiling) as solid 'inside' (false in CGAL's binary cut map)  
    // This stops the empty interior space from leaking infinitely upward  
    external\_nodes \= false;   
      
    // Preset the remaining bounding box faces as empty 'outside' space (true)  
    external\_nodes \= true;  
    external\_nodes \= true;  
    external\_nodes \= true;  
    external\_nodes \= true;  
    external\_nodes \= true;

    std::vector\<Point\_3\> reconstructed\_vertices;  
    std::vector\<std::vector\<std::size\_t\>\> reconstructed\_facets;

    std::cout \<\< "Running graph-cut optimization..." \<\< std::endl;  
    ksr.reconstruct(  
        lambda,   
        external\_nodes,   
        std::back\_inserter(reconstructed\_vertices),   
        std::back\_inserter(reconstructed\_facets)  
    );

    // 8\. Export the reconstructed mesh  
    if (reconstructed\_facets.empty()) {  
        std::cerr \<\< "Reconstruction failed: Output facet list is empty." \<\< std::endl;  
        return EXIT\_FAILURE;  
    }

    std::cout \<\< "Reconstruction complete. Facets: " \<\< reconstructed\_facets.size() \<\< std::endl;  
    if (CGAL::IO::write\_polygon\_soup(output\_path, reconstructed\_vertices, reconstructed\_facets)) {  
        std::cout \<\< "Successfully saved watertight ceiling mesh to: " \<\< output\_path \<\< std::endl;  
    } else {  
        std::cerr \<\< "Error: Failed to write mesh to: " \<\< output\_path \<\< std::endl;  
        return EXIT\_FAILURE;  
    }

    return EXIT\_SUCCESS;  
}

## **Energy Minimization and Dual Graph Optimization**

Once the bounding box of the building scan is partitioned into a set of convex polyhedral cells ![][image14] by the kinetic simulation, the final watertight surface is extracted by assigning a binary label ![][image15] to each polyhedron ![][image16].8 A label of ![][image9] indicates occupied space (solid structure, such as walls and ceilings), while a label of ![][image10] represents empty air inside the room.8 The final surface is defined as the set of boundary facets separating these differently labeled volumes.8

### **Structured Formulation of the Normalized Energy Function**

The optimal labeling configuration ![][image17] is found by minimizing a two-term energy function using a graph-cut solver 1:  
![][image18]  
The parameter $\\lambda \\in  
The data term ![][image19] measures the coherence of the assigned volume labels with the oriented normals of the inlier points 8:  
![][image20]  
where ![][image21] is the set of inlier points projected onto the facets of polyhedron ![][image4], and ![][image22] is a voting function defined by the normal direction 1:  
![][image23]  
![][image24]  
The vector ![][image25] is the oriented normal of the inlier point ![][image26], and ![][image27] is the directional vector from ![][image26] toward the center of mass of polyhedron ![][image4]. If the normal is oriented outward (away from the interior volume), it votes for the cell to be labeled as inside (occupied structure).1  
The complexity term ![][image28] penalizes configurations with large boundary surface areas to avoid jagged, physically implausible transitions :  
![][image29]  
where ![][image30] represents adjacent polyhedra, ![][image31] is the area of their shared facet, and ![][image32] is the total area of all facets within the partition.10 The term ![][image33] is the Kronecker delta, which evaluates to ![][image10] when adjacent cells share the same label, and ![][image9] otherwise.10  
The normalization factor ![][image34] scales the regularization term by twice the total number of inliers ![][image35] relative to the total area ![][image32] of all faces in the partition.10 This scaling ensures that the data term (which scales with the point count) and the complexity term (which scales with facet area) remain balanced, independent of the absolute size or sampling density of the point cloud.1

### **Boundary Mapping and External Node Constraints**

To run the graph-cut optimization, the polyhedral partition is represented as a dual graph.8 Each cell is mapped to a vertex, and adjacent cells are connected by edges weighted according to their shared facet area and point votes.8  
Because cells along the boundary of the bounding box lack adjacent cells on their outer faces, the dual graph is augmented with six external vertices representing the faces of the bounding box.8 The external\_nodes parameter allows the user to pre-assign labels to these boundary faces or leave them to be optimized by the solver.8  
For architectural reconstructions, configuring these boundary nodes is critical 8:

* **Aerial Building Scans:** The bottom face (ZMIN) represents the solid ground and is preset to inside.8 The other five boundary faces are preset to outside.8 In CGAL, the helper method reconstruct\_with\_ground automates this setup by fitting a ground plane to the lowest detected horizontal features and linking the below-ground boundary cells to the inside external node.8  
* **Interior Room Scans:** Because the scanner operates from inside the room, the ceiling plane separates the empty interior from the solid structure above.8 To prevent the interior empty space label from leaking through the ceiling, the top boundary face (ZMAX) is mapped to inside, while the floor and wall boundaries are preset to outside.8 This boundary constraint forces the graph-cut solver to terminate the room volume cleanly along the detected ceiling plane.8

To compare these configuration strategies, the boundary conditions for different scan orientations are detailed in the table below:

| Scan Orientation | Bounding Face ZMIN | Bounding Face ZMAX | Lateral Bounding Faces (X/Y) | Recommended Solver Method |
| :---- | :---- | :---- | :---- | :---- |
| **Aerial LiDAR (Roofs)** 8 | inside (solid ground) | outside (empty air) | outside (empty air) | reconstruct\_with\_ground() 10 |
| **Interior Room Scan** 8 | outside (empty floor) | inside (solid structure) | outside (empty walls) | reconstruct() (with manual mapping) 15 |
| **Inverted Scan (Apartment)** 8 | inside (solid structure) | inside (solid structure) | inside (solid structure) | reconstruct() (with all boundary faces set to inside) 8 |

## **Computational Benchmarks and Parameter Calibration**

The computational performance and reconstruction quality of the kinetic shape reconstruction pipeline are determined by the interaction of its geometric parameters.

### **Performance Evaluation under Spatial Subdivision**

The kinetic space partitioner supports an adaptive octree subdivision to accelerate the construction of the polyhedral complex.5 This decomposition limits the propagation of planar shapes to the octree nodes they initially intersect.7 Once all sub-volumes are processed independently, they are merged into a single conformal partition.5 This spatial subdivision significantly improves performance and scalability.5  
The table below illustrates the performance and memory savings achieved using this spatial subdivision strategy across different model complexities :

| Dataset Model | Input Points | Planar Shapes | Output Facets | Partitioning Time | Labeling Time | Memory Peak |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Meeting Room** | **![][image36]** | **![][image37]** | **![][image38]** | **![][image39]** (with subdivision) | ![][image40] | ![][image41] |
| **Full Thing** | **![][image42]** | **![][image43]** | **![][image44]** | **![][image45]** (with subdivision) | ![][image46] | ![][image47] |
| **Tower of Pi** | **![][image48]** | **![][image49]** | **![][image50]** | **![][image51]** (with subdivision) | ![][image52] | ![][image53] |
| **Temple** | **![][image54]** | **![][image55]** | **![][image56]** | **![][image57]** (no subdivision) | ![][image58] | ![][image59] |
| **Church (Aerial)** | **![][image60]** | **![][image61]** | **![][image62]** | **![][image63]** (with subdivision) | ![][image64] | ![][image65] |

### **Parameter Tuning Matrix**

To handle common structural anomalies encountered during ceiling reconstruction, parameters should be calibrated according to the diagnostic matrix below:

| Structural Anomaly | Root Cause | Primary Parameter | Recommended Action |
| :---- | :---- | :---- | :---- |
| **Ceiling Plane Disappears** 8 | The detected ceiling plane has too few inliers, or the graph-cut solver favors a simpler volume with no ceiling.8 | minimum\_region\_size (![][image66]) 12 graphcut\_lambda (![][image67]) 10 | • Decrease ![][image66] to retain smaller ceiling fragments.12 • Decrease ![][image67] (e.g., to ![][image68]) to prioritize data faithfulness over simplification.8 |
| **Jagged or Fractured Ceiling Plane** 1 | Structural fixtures (beams, HVAC) segment the ceiling into multiple parallel planes.1 | regularize\_coplanarity 14 maximum\_offset 12 | • Enable regularize\_coplanarity.14 • Increase the maximum\_offset distance to merge offset horizontal segments.12 |
| **Ceiling Bleeds Infinitely Upward** 8 | Empty space labels "leak" past the ceiling because of a lack of points above it.8 | external\_nodes mapping 10 | • Map the ZMAX face of the bounding box to false (inside/occupied).8 This establishes an upper boundary that stops the leak.10 |
| **Ceiling Fails to Connect to Walls** 1 | Occlusions along the ceiling-wall joints prevent the planar polygons from intersecting.1 | k\_intersections (![][image69]) 7 bbox\_dilation\_ratio 16 | • Increase ![][image69] to ![][image70] or ![][image71] to allow the ceiling polygon to propagate further.7 • Increase bbox\_dilation\_ratio to expand the outer boundaries.7 |
| **Extreme Processing Latency** 1 | Large numbers of small, noisy planar shapes create an overly complex partition.1 | max\_octree\_depth 16 minimum\_region\_size (![][image66]) 12 | • Enable octree subdivision by setting max\_octree\_depth to ![][image70] or ![][image71].5 • Increase ![][image66] to filter out small clutter (e.g., furniture, decor).1 |

## **Conclusions**

Implementing the Kinetic Shape Reconstruction framework offers a robust, scalable solution for modeling building ceilings from noisy, incomplete point cloud data.1 By replacing global plane arrangements with a localized, kinetic propagation model, the algorithm achieves sub-cubic spatial complexity, making it highly effective for large-scale architectural reconstructions.1  
For developers using AI-assisted environments like Cursor, leveraging CGAL's exact computation kernels ensures numerical stability during the partitioning simulation.1 Additionally, configuring boundary conditions with the external\_nodes parameter allows the graph-cut solver to robustly handle the inverted orientation of interior scans, preventing spatial leakage and producing accurate, watertight ceiling models.8

#### **Works cited**

1. tog2020.pdf  
2. Kinetic Expansion of Linear Structural Elements: A Hybrid Method for Floorplan Reconstruction From Indoor Scene Point Cloud \- IEEE Xplore, accessed June 13, 2026, [https://ieeexplore.ieee.org/iel8/4609443/10766875/11184404.pdf](https://ieeexplore.ieee.org/iel8/4609443/10766875/11184404.pdf)  
3. Unified Primitive Proxies for Structured Shape Completion \- arXiv, accessed June 13, 2026, [https://arxiv.org/html/2601.00759v2](https://arxiv.org/html/2601.00759v2)  
4. Structure-preserving Planar Simplification for Indoor Environments \- ResearchGate, accessed June 13, 2026, [https://www.researchgate.net/publication/383090656\_Structure-preserving\_Planar\_Simplification\_for\_Indoor\_Environments](https://www.researchgate.net/publication/383090656_Structure-preserving_Planar_Simplification_for_Indoor_Environments)  
5. New in CGAL: Kinetic Space Partition and Kinetic Surface Reconstruction, accessed June 13, 2026, [https://www.cgal.org/2024/05/29/Kinetic\_surface\_reconstruction/](https://www.cgal.org/2024/05/29/Kinetic_surface_reconstruction/)  
6. lhq1630798/KSR-imp \- GitHub, accessed June 13, 2026, [https://github.com/lhq1630798/KSR-imp](https://github.com/lhq1630798/KSR-imp)  
7. CGAL 6.1.1 \- Kinetic Space Partition: User Manual, accessed June 13, 2026, [https://doc.cgal.org/latest/Kinetic\_space\_partition/index.html](https://doc.cgal.org/latest/Kinetic_space_partition/index.html)  
8. CGAL 6.1 \- Kinetic Surface Reconstruction: User Manual \- TAU, accessed June 13, 2026, [https://www.cs.tau.ac.il/\~efif/doc\_output10/Kinetic\_surface\_reconstruction/](https://www.cs.tau.ac.il/~efif/doc_output10/Kinetic_surface_reconstruction/)  
9. Geometry and Topology Reconstruction of BIM Wall Objects from Photogrammetric Meshes and Laser Point Clouds \- MDPI, accessed June 13, 2026, [https://www.mdpi.com/2072-4292/15/11/2856](https://www.mdpi.com/2072-4292/15/11/2856)  
10. CGAL 6.2 \- Kinetic Surface Reconstruction: User Manual, accessed June 13, 2026, [https://doc.cgal.org/latest/Kinetic\_surface\_reconstruction/index.html](https://doc.cgal.org/latest/Kinetic_surface_reconstruction/index.html)  
11. SimpliCity: Reconstructing Buildings with Simple Regularized 3D Models \- CVF Open Access, accessed June 13, 2026, [https://openaccess.thecvf.com/content/CVPR2024W/USM/papers/Bauchet\_SimpliCity\_Reconstructing\_Buildings\_with\_Simple\_Regularized\_3D\_Models\_CVPRW\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024W/USM/papers/Bauchet_SimpliCity_Reconstructing_Buildings_with_Simple_Regularized_3D_Models_CVPRW_2024_paper.pdf)  
12. Kinetic Surface Reconstruction: CGAL::Kinetic\_surface\_reconstruction\_3\< GeomTraits, PointRange, PointMap, NormalMap, IntersectionKernel \> Class Template Reference, accessed June 13, 2026, [https://www.cs.tau.ac.il/\~efif/doc\_output10/Kinetic\_surface\_reconstruction/classCGAL\_1\_1Kinetic\_\_surface\_\_reconstruction\_\_3.html](https://www.cs.tau.ac.il/~efif/doc_output10/Kinetic_surface_reconstruction/classCGAL_1_1Kinetic__surface__reconstruction__3.html)  
13. Kinetic Surface Reconstruction: CGAL::Kinetic\_surface\_reconstruction\_3\< GeomTraits, PointRange, PointMap, NormalMap, IntersectionKernel \> Class Template Reference, accessed June 13, 2026, [https://doc.cgal.org/latest/Kinetic\_surface\_reconstruction/classCGAL\_1\_1Kinetic\_\_surface\_\_reconstruction\_\_3.html](https://doc.cgal.org/latest/Kinetic_surface_reconstruction/classCGAL_1_1Kinetic__surface__reconstruction__3.html)  
14. CGAL 6.1.1 \- Kinetic Surface Reconstruction ... \- CGAL manual, accessed June 13, 2026, [https://doc.cgal.org/latest/Kinetic\_surface\_reconstruction/Kinetic\_surface\_reconstruction\_2ksr\_building\_8cpp-example.html](https://doc.cgal.org/latest/Kinetic_surface_reconstruction/Kinetic_surface_reconstruction_2ksr_building_8cpp-example.html)  
15. CGAL 6.1 \- Kinetic Surface Reconstruction: Kinetic\_surface\_reconstruction/ksr\_parameters.cpp, accessed June 13, 2026, [https://www.cs.tau.ac.il/\~efif/doc\_output10/Kinetic\_surface\_reconstruction/Kinetic\_surface\_reconstruction\_2ksr\_parameters\_8cpp-example.html](https://www.cs.tau.ac.il/~efif/doc_output10/Kinetic_surface_reconstruction/Kinetic_surface_reconstruction_2ksr_parameters_8cpp-example.html)  
16. Kinetic Space Partition: CGAL::Kinetic\_space\_partition\_3\< GeomTraits, IntersectionTraits \> Class Template Reference, accessed June 13, 2026, [https://doc.cgal.org/latest/Kinetic\_space\_partition/classCGAL\_1\_1Kinetic\_\_space\_\_partition\_\_3.html](https://doc.cgal.org/latest/Kinetic_space_partition/classCGAL_1_1Kinetic__space__partition__3.html)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA3klEQVR4XmNgGAWUgnlA/BmI/0PxAhRZCPjLgJAHYWdUaUyArBgb2AfEKuiC2AAjEG8H4vUMEMOCUKXBAJclGCAfiE2gbFyu+4MugAu8RWJ/YIAYxockpgbEnUh8vADZJaBwAfFvIoktA2IeJD5OAAqvzWhi6F7F5m2sADm8kMVABnRD+b+Q5PCCd+gCUABznTYQt6DJ4QS4vLCbASJ3D4g50eSwAhYg3osuCAVMDJhhhxMwA/EbID6JLoEEvgHxd3RBdLAKiD8yQNIXKF2B8h42oA/E2eiCo2AUDGkAAMruNN36aWNMAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAXCAYAAACrggdNAAACk0lEQVR4Xu2WOYtVQRCFyx1RVFA0EEFBVBCXASMXRFwCI1cwFjExEAMTETQwEWF+goyJgQgGgoIbuOCGiRgJziSiiLiLC+517Kr36p7p5r15ywyIHxzoOtW3l3u7b7fIf0aMmarPqg+qg5QbNlaz0Sa/VGOs/Fu1I+TAIoqLbFDNsPIoVa/qkGp6rUYevFHU7yQvVeOsjEntDznwQLWEvEG8Vp2yMj453hQacx2wHHNDtZZN46Tqo9Tb6KtkEz+l2s/6alrmSxpPDtSfxCaYJinps0Z5Uz0te8yDmHmS95k46BzXJA2eOSrppW0m38E4P7EJ0NEWKw+oHoWc419tFvlvVVvJY7AsL6rOSWpjWzX9l9JkHeTPsmkgNzsaL1TvrLxUUoXc3sCyRG4n+Y0GA7AXVli59LV+UDxB9TjE9yT/HMDeuuXBXEkVp1p8XjXHkwT2E+quDN5u8xqBvergBeKZKcFboDoeYrBGqm1/U10KcWSXhLrvY9AANMh176uekZcjPod9gzh+hdOqySF2LqiuqO6qHlKOqfVRWgo5fE+xd5M8BksZKyDC/XK7rVCZFAbWDKh7JOP5EVAi7qfo4dkTFmNptcuQv9RhydeD18cm8YYNw/terDpGuVYY0qRwoqOO3zIiX1XX2SRK7V+WlMMRMpFyrVDrp9+CJ/VcBZ/QKk4Y2LxP2QyMVV1l0xgtzb3UZqm1Exv+olpoPs4I/JHg5/5Kzl4pDwoX0VeSzpcSuC+i33bZLjSO8arvZkbx/atEblJnJB0XOJ9wLuFul2OZah+bLXBbdYfNdsCkNrI5zGAMfpvvCMul+WOhG2C/d+JIGMRzSZMbCbC08UPqCqV9003wZ+1hs9OsY6PLxMv1v8kfibCwgvM0KBAAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAcCAYAAACtQ6WLAAAAe0lEQVR4XmNgGOTAEYht0QVh4D8Qr0UXhAGQZAG6IAjoM0AkmZAFbYDYC4h3QyV9oXwwKALiEqjEWygfhFEASDIXXRAEdBkgkozoEiCwhgEiiRW8ZsAjCZLYhMTfhsQGS1pA2RlIbDAASYK8UwfEK5AlYADkeQl0wREPAGL/GMEfWDMiAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAaCAYAAAB7GkaWAAAAZ0lEQVR4XmNgGHigAMT30QVh4C0Q/0cXpAx0AnECuiAI/IDSIPsckSVmAjETlA2SdEWSY6iF0v0MeFwKkriALggCIgwQSTF0CRA4z4AwshyIpZHkwBI7oOxXyBIg4MwAUfAHXWLEAwCaDRQuuqoUtAAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAaCAYAAADMp76xAAACF0lEQVR4Xu2WP0hWURjG3/wziFAIokNKZAQOhZAg6tLSohFCLo4trqVtNri5uLWIjiZu6iKEEDQ0ieJQNGgitIWV0RI1ZPk8nPPh+R7PuffzGoXx/eDhuzzP+733nHPP/WNW5fzTpUaCETWKcBkag+agjsDvD46zuAU9VzMB+79Ws1IWoN/QLnQXug7NQB+gPp/lUQ/9VDOHp+YW51RwML+gSxqACXP5Gw0iHELdanp45VKTTvlRuCJ5f2B+X02h0bL7LFs659WtaGt8NdekQQMhdaKQt9CamgHs8UVND69s7jlumit6p0GE3Gbmau6oCQagQXP5M3/cWlbhYH5RzRDuNxbF9m0RYpNqhh5Ds+ZyHlPcPgpz3i9JWBA7SRFqLbtX1v4twXxezZCzDJh7P+SqZfdi9llN4Tu0rmaJ0orsaxBBB3IBGhbvip2sC2H2SE3hG7SlZkglK9wLPVAzQarXDXNZjQYCa1bUDNkzV8TVjkH/o3h85R5AdeKT1ICXrDx7FRyHsOaJmgqL+OLQQfN74JN4LdA1c5dtSDLCXrfVBJt2POAec0+MGKxpUjPGCzveHtxH/B0tqygntZLvoVU1Pbyh+L9pDTx5b8nC8CWwo6aHq1f0pFx1Pvr+OD+gduilBh5+QHWqWQFFJ5rLOLQBtWng4WQ4qdMwCU2p+Te5By2qmYDfFNtq/gtiH0ExHqpRpcr/yBH3+HksRzM6hQAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAAgklEQVR4XmNgGLqADYjvowtiAyuB+D+6IG0BMxAvAuIAdAlkIA3Ej6BskPvakeRQALLjQeyrSHwUoIrEBimMQ+JjBc4MRAbNLwYiFYIULUQXRAfcDBCFcugS6KCDAY+1yLr/APFfJD4KAJnQCMS2ULYMqjQCgCSPA/FnIOZBkxueAACBrBmZg6ILPgAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAaCAYAAAADiYpyAAACpklEQVR4Xu2XS6hPURTGlzcR4UZILiZEKeoaSilRIhkYm5hRjFAYUBiIGGCGDMhj4JEMTdy6SSJRSMrMM+SR1/pae7v7fu3Huf97/e//6vzq65zzrb3P2fvss9feR6SmZrCziI0EG9n4n1isusFmgjmq+2xW5Yjqg+q30xfVW/Iu/C3dXEaofrBZ4KjqJJu9wXeamSXm3+JAE/ipWsKmY4bE2wtSfiVQ+Q6bjtRL+peMlfwzL0k6flYanCJIMrjpSg4oY2RgXsRD1U02A9AeTOEYE6TB9j6SdMUrYrF1gbdetSu43qnaFlz3B3jmCjaVVarVYvEz7nxqjxIG4uPZLJEa8eViPhKQZ4Fqn+q4i0GjVYdVz4JyfSXWnjaxF35CLI5zCNOIQXwHmyV8h96r3qm+uusHqslBOfDJHc+LlRnuhPNTvlAfGSbxF+HJ5QcP4qfZzOHzwyYOJMA0AKjzLfDHBecAo/eSvBj3xL6okNmS7yhir9kksA3oZDPHE8k/NAXq7GUzYIqkl76QA2xI95KdArGtbBKfVXfZzIGb5h4aY7pYnVEc6EdSbVooFhvKAQJlLrOZAxWeslnAJ6sUSJq/2CT2qK6JJd0YqftflJ6x28F5CMr4aVwESyAqbOZAAdTBHIyxVizZfVdNophnrmqa5JMi/GVsKl3SXadDbFBioMxENpljqo9iKwQ2JZhPpREMwUM2sEmkOhiyRXWdTccL1VU2HRgE3P8QBxylXWnTOCjVfnyQ+Zey6cBoN9oZfCVYYgcc34Hn7jhE4l9cqaOoM4/NCpTu2zTw2479wcjAeyOWFzzI+qUGzxTb3PWG3ar9bLYS7e6IXSjYLtWWtzWqc2wmwD/HYzZbjVfuiK+g3R2rEvv5ioHkO2jAlJjPZk1NTcvwBzg4o1h3Tku5AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAApElEQVR4XmNgGJpAHl0AHZwA4qtA7AbEj4H4AIosFMwF4r9oYv+BuBRNDKvgDKg4HEhDBTyRBYEgByoOBwlQAVNkQSCIgIqrwgQqoQL6MAEoCIaKw22qggqgKwyCioejCxjDBKAgDCoON8AOKmAJE4CCWKg4yLNgwA4VAJmADGBOQgEggUloYtug4igAm24QH+R+DLCcARKNIBqkqABVehRQAwAA4fYow14SzbMAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAZklEQVR4XmNgGLqAGV0AGYAka4D4PxBnocnBwQ0gXgfEfgwEFCKDoaIwB10QGwApzEUXxAZACvPQBbEBkMICdEFsAKSwEF0QHYgwQBT2oEvAwGogfg3ET4D4MZR+CcS/kBWNAsoBAO7yGbPFo+KDAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAWCAYAAACWl1FwAAAAjklEQVR4Xu3WsQ1BYRiF4UMhDCCxgkKrVNjCDNa4xhBjMIBSYwBhA4mWhPfmVk4viu88ydv8p/urT4qIiPiXuT+ENKAbHan3PUWfznSlkW2BPT1o4kNIO3rRzIeQNvSmhQ+VrdV9ysqHihp1n7H0oaItPWnqQ0UHutPYh2rag+1EFxraVlZ75ueSjYj4gQ914hHWBkKn9wAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEoAAAAWCAYAAABnnAr9AAAAjUlEQVR4Xu3ULQpCQQBF4edPUBCLYHED7sV1uCdxF5pdgQg2qxgMFsFg0uB5iOFd5mXL+eCUuXVmqkqSJLVb5IGa1vSieQ762tGDpjmoqvp0ogsNYxPGdKMDdWMTZvSkbQ5qqj/oN61yUNnvZm1yUNmIrrSnTmwq6NGRzjSITS3q53inSQ4qW+aBJOm/PpY4EiDzUzbSAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAZCAYAAADNAiUZAAABRUlEQVR4Xu2UTytFQRjGX5KSupRkacvipsgX4BPYWCkLKfkCd8UXsLGUfAfZUhZKSZT/VpTsLSis8DzNO/XOa6Z7usvb+dXTnec3Z+bcU+eMSE0304+cIL/IoZuzbCDvyCey4ubIDnKGfCGTbi5hWsLNhrTPavc8IEem3yGnpi8hg6ZvmvE/eIPrjLs1vaHOQzesYz6hZ8ILMiph4a7zl+ojV65H6PZ0vIr0mbl1M05YlrBw2/lj9RGOSze1/gDZR56QOeMTxiX/pC/qR7T7zSMl3xYuusk4hi+V7Z6Sb8u8hIX8bEhLwktE16uutHnJV2JMwvd5jzSRZ0k3K21e8h3BjX5M/1DnoXv0sgq5f8u+YPqiOg/djJdV4EIea5EL5M30CK9bM31LXUdMSVj8qr88g3MMSJg/l3CCfSM9yRU1NTVdwR8yvl9iY8MEPgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAZCAYAAAAFbs/PAAAAmklEQVR4XmNgGFJAGoibgHgCELOiyaEAMyD+j4a/oqhAAjuB+B6aGEjDSTQxMDgCxM/QxHQYIBr40MQZbKAS6OAHEF9HFwQBkOLD6IJAIIcuAALiDBANoFAhClxiwO4cnAAWdEQDqmuwQxf4ywDRUIguAQSPgPgMuiAsDkA4ESrGAcT3gfgqTBE6mMGA0ATCf4DYEkXFKBhUAABG+yf8VXiXsAAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFgAAAAaCAYAAAAzBZtTAAAC4klEQVR4Xu2YS6hNURjHP2XARERIuvIoJCmUga7rOWBA3vLo5tUlSgxIHleMJJmYiKGJgYmJgUKJASXKayZJTFDekfj+rX2y7t9ad39nreOcm/av/nX2/1tr7W99Z++91t4iFRUV/55RqrVsNpFlqjFspjBMNYXNFrJa9Ut1WTWWYs1kruq6uFwW9QzZWKL6JG6AGxRrJcjnJJseR1QfVF9UWylWL/1V39kkzovLKRl0XshmC0E++PNDPFFd844fqW57x1aeiTtPTb2xRcrbRNksGZ0DDFRtU+30VC/IZz6byiAJ5wpvMJtGcCeExvTZJOVtoryQjM4e68WNg4JOVbWpRqqG+o2MYJwONpUHEs4V3gU2jVgKvE7K20RBxzts1sl91S42M4gVGH5oojHfgqXAtUU3CXRMWiELZqiOs5kJcprNpsQLGfMtWAq8XMrbBOmUxI4eP9jIpEviOcUKGfMtWArcT1ybFRwo47nEB3/PRgQUeGaJLExTXRGXzwSK1YgVMuZbsBQYTBbX7mLx2wQ63GKzYA8bEVDgpSWysFh1T9x4QyhWI1bImG/BWuDhqq/iXjwWUCwKBs7d//5kI5MDEp/wRwnH4D1l04i1wGiDnZIZf2+H597E4ne36rHYb4OVqn1sZoK8QovcGgkXAx4W2xp4Zh72jnvDUuCkRW6v/OnkX4WdqtOqM55XBm5tfBhpFMhrDpsFiO3wjk8Vns/bwttPfgi8bnN/Jnmb9llcx1nkpwx2SPVNXDIjVAN6husC5+9gswBviojfVT0Ud05csT5YMFFkvETFwOPmteploVeqd6rxfqOCrBcNBgtMznN1urgr55i4/fEJ+bsAZWAy89hM4CobiWyUBhb4pmqD6ij5zQSTwa4iB3xjGc1mIlkfe5h2cYtc7BZtBpjMQTbr5A0bGZyVBha4L9AtbkLnJO1jUaOYpLokLpdVFPsvwE5iN5tNZLtqHJsVFRUVFX2b3+IFvjAF2ofUAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAaCAYAAADMp76xAAABdklEQVR4Xu2VsS8FQRDGP0KBhIJIEAmVRqEQvQiViIRCoVBIRKdTiYRKRSVE4x+gVmn8AbSqFxqJRCMKhWAms/fe3tibe0ueiOwv+bKXb3Zv5u3O7QMSiT9PE2mZdEoay4e+MkSqaPMXeSF9kN7dyDJ5Qh2TGsA4JG+X5z07r2HwMa57GsmHCxlAuDD2LrX5U3ohLz6E7NIwqY/UQ2r15lnw+l3lzTvfZI+0ok2DBdKVNiM5QLgw/tEmr27kxZN+wOBBG9+A891qs4xjUrN75hdMe7Ei+Ag7tRlJ1k5LOlDGlhv3ET6eEHeQnrXkf/Eh1iD5BnWgXnjxjTYLeCPNlai/OjvMGSQn73Q0/FXHLD4itWkzkhPYLdEOY/evUWuHTcjdWEZFG5HMQHKG2nAWYb8KBy/c86MfMFgknWszkqzge1KL81adZ97jU5BJ3JsxjELWbUP+NDry4VK6USs6E99aDYdvhA1I4TuQa28iNyORSCQS/4JPOfFNGL32Qm0AAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAAAqElEQVR4XmNgGBZABIi10QUJAS8g/gLE/4F4P5pcPBA7oolhBSDNLlC2MRB/BWIjBoiLfgOxFFQOAyQyQDSjg1dAfA9dEB08ZEDVrAfE/xggNoNsBMkJI8mjAJDkMXRBIIgDYlt0QXQA0uyKLkgMAIUoNv8SBe4zUKAZpPEwuiCxADl+SQKxDAgnpwOxOpIcQVDIgND8F1mCWABKhiADzNElRsEooAgAAOPAH5vL6sqUAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABOCAYAAACdbkoxAAAKK0lEQVR4Xu3dCcw8yRjH8Yd134Kw6/oTWdYZt13Cn3VfKywRCRtBhMSRECzrDisiiCNiHftaYcUZR8giWfdNXMkGIcu673Xf1C9V5a153qd6Zt6Z6Zl35/tJKu/U011d/c68b3dNdXW1GQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACM48ouf1JKd3YxHEz/9YHkyOb1dVO6bJMfW7R/63CEDzQe5ANL9C8fSK6V0hXKa9X95mbZGI7xgSX6jg8kl0/plk1+U/4mAGCj/MkmD5D/SekWlhtx/2ziiD0tpXf64Ar81fLnNJRe9v+1s1+kdBUX+7vtPSH6/Fii/ave6gMrpvfgLz7Y0P9FzwNT+oftfg6/S+n3Tf6Fu6tO+HdKF3axX1suc6cm9qXm9Rg+YHkfjvcLCjUy6++m48cpJa7Psv17PK/Eq1Ntd91K75XW/VsTu46tttEIAAeSDo7tCVsNtXMsfxNWYwTDnp7SB31wRdQL1Gtc/Tmlmzf5r6T0giZfPdL2buOoIDYGv3/XSOmMlB5v4+9PbWyc7BcUD7bhBt1Jlsvf1S+wOK7/s7u4WBX97kN193zXB+agL2vRflS+kdVSgy/yBx8otK27BbELuRgAbLW32OSBWQfbe6R0dEoPaeJYv6da/yT6LpfvrfdDi5cpdtgHV+hMH2iohzfax1VTnUP1atklfLBQ46hX9km2d5nvfar0BSrq2R6qu8fXOY+HWr/83S0ve5ZfUNzUByz3mLWXPavoC4R8PqVP+CAAbDMdLHU5p6VLPNHB9aC4nA9cQJyf0leb/M1SekR5/e0mLtFJUBT/nA8m37R8khxLb/9kXQ22HRuuV8t0WS+iZb2yt7K9y3y+2knpxT5ow3X39OqYlcpHY/s+a3nZRf0Cy2MiI7rUGznX4v3UMSiKA8DW0kHRX44Y2+m2e3DWSUCv69ie21nuPVJMY3707b09kOv1jS1fPtHr69v+Gh7PsFz+tJR+VGIfKzH1eFw6pRNL/ppluXoiv1xiopPVG0r+JbY7Bkj515bXi9B27pfSsZZP6r0Tmm4ieJsPFiqj38U7zvrbWzbt31Bd62qwieq9lA8Wz7T+fin+PB8sXmOT5e7t8q1efKjunnnX9zQm8o8+aHm7vW2f6wNFb33FX+SDRa8MAGylRQ6KajxpcLhPusy6Y/nutjdabsQMeYJN7ode+14gxa5aXp9VfmpckAY9V1onuutuVtGJSPmLufx7m3yN+fzjmrx6xvw687qN5W2oofZyyyfS3jZ1h29voHuvzNWsv6zyn3P7eZ9u+fPWZ329WqBD+zdU1zobbN9L6cM+WAxdJlS8d8lSN3roy0b1fBveTmSo7p55149E21DsIz5YROvLULzXQO6VAYCto+kDdDLZFDrp/9zygVqNnFZ08L61TZ4Itc5+BmdXKu+nHfD1Kv/JIDaU/1kQm5dvoKlH7GHltXojP9osUwNYvYGeLp9+2gcbi+7jrHwD3Zu1waZ1Zkmz0rqHys+IxmFFy4YaU2rEadlFmth7SsxTgzeKS6/uyv/OveT/vnvUo1f/F/1dqopFl0OvndI7fLDo7XsvLkPLAGCr6ATR64kZU+1x0QFf9HqWBpsorrsj611+vV6OWaj8tLFgyn8qiA3lzwti81L53jZ0R2jbILhvSk9u8tUPrD9Vg3oRe9tfNu3fUF2zNtiWSfWd0LyO/i/qYHtPvXK+EV/pPfdldKnRx0Sx3heoXt1D5l2/0v7VL0J+GhgNP+ht9/0pXd0Hi6jMpnyBAICNpwOiLkfsd5D+Yywf0Kel3u3/lfajjhureU0BoAk121jPoy2PQVuU6vhGEPN5f5KJ1mkt2mCr4/p8T0flbxrRJclXuZjUfTh2Ippp7N+0ffSfay/dvxbo0P4N1TV2g+1NKb2iyavuqP7HWhxXLJqiow6cv6OL6++1tx2N59K0Jl6v7iHzri/qrW3LqRGr/O1L/kYlHxn6P4/KfN92x8+ql9KLygDAVqoHxPay4jpoP3TZsKpzQB1uYr2Dt+JfTOnslF5v8ZQCs9K21Liq1Jj19Srf9sJFPVPKt+Pe6iSq+6WGmsrX2e+roZN4dPKs60ZlXm1xfFWG6hoalL9MR1iuRw2Hli61R/V/K6Wvu5gaMn5dxTR9Su/GD/FlRDHt02/8AovrniaqY8i7LS6jWBvXazXwq8vY3i8NXm+7Es0XdyWLywDAVqon/Nv6BSPTAb+O0dLBWz1KarRpTrjnWJ4R/8eWG3W+IVJPJm1SA2leunlBjTXVoxOmetpUn2K/TOkOlvejrqN906VRjfP5SSmjOzjrOiqry2TajtZXTJPbzksnQs2y739HNbL1HvXG7EUnO90Aorh/FJloW6f64ApF+yf1s9b79VPb32c5K90g4f+eKj1Zop2IWLTPujO4amf812ek91Db03jCKzbrRaLfXzfaKK47nj1f9yyiOoZoff2de/exyW3VaUpqmqUhuWN7n+qgnl6VV4+jp5tr9vP/AgDYQNG3+nr5cNvpPZjWaGiN/Z7pKRrz7N+61WljlqV3J2rPfureT5lV0VQyvek7Imr83sQHAQAH09ds7yBnzRS/SSeqdal3J85CPVkP8MERzLp/m0D7esgHF6RGySzUcDnkgweQ3sOo9zCi+Q0BABcguhSpy1BnWb5MqMuX0ZQDi4gGRM/jsA+MRJe8z/TBwId8YCSz7t8mWMVzdd9u+T2Y5iA1bIdoDKYuNU/zcR8AAGwXTdWgk9+RfkGHJjjVSbWlGwDOSeniLt7yY59EJ6sb+uAIpvVoPNEHRjZt/zZBdNfmsugO7fZuaG+Vda/L0Gfu76gFAGypeXorXmeTz3XUjQm6i09629FNEr3HUEVTPwAAAKARPcNSl1H1rf6Q5cl3W7qrUr1sVVt2aExTW6ZV55ySh1seO3ZUSic3cQAAgK32BcsNpeozzeuIHjr/lCbfNth+Zf1pMHoNtnZgv+aLu2d5rRslAAAAYHt71y7p8i09p1NzqbXa8rq5QQ9hj/QabCpfe9n0PMnq6OY1AADAVvMNtnu5vKc72nQZtWrLa6LU3pi0XoOt92gpAAAAFDs2+VDvaTO0n5HSc5u8H8NWqfHW6jXY6iVQAAAAdJyf0ikupkdH3cDyRLOeHl3UNr70aB01zl5pk4/w0VxwdaoQPU7ntyX5iVGPd3kAAAAsSA9Ef6kPBp7tAx3H+QAAAAAWc4zlB3xPc4IPBN7nAwAAAFieRWebf5TN/nQFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABwgP0Pg5iSvndfXJoAAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAaCAYAAADFTB7LAAAB1klEQVR4Xu2WSytFURTH/54DlExImUgyMfAJRMJIychIZuZKlAHKyMjEY2jgQ8hIko9AIm6SjDzKIyH2uuvsY/c/+zzu7dId3F+tbue31tpn33P2PucAFcqTdhPVLIk+Fv9Fs4kbltAJ95D7puNU1kw8QRsl3kzckzsNq/34TnqF336XThMP5DLhG8zyjvjchYlZlgHX8Pe9mBhjmYYMdMgyoAmaXyYv+CZgkdwxS0M3kvsiTEIbBjnh4LvCGyaeyblI/SjLAMk1sIzjBNGTM74JyvEqOcs0ovUukpP1nwnfyRlfjRzL7fKRQ7TeZR+6tjMhAx2wdBiB1vBuFldLziK5I5YO20j+AyF2/Q2QdzmH1rSSTzqB5IZYOswjuT/kDOmFkv9iifi+KcTnLHNIr8kjRUmFdgP5XmPia1gaLpE8prCF9Jo8UhT3/FuB5ns5ESC5LpZQb59/cZPYM/HKklmEDjBMfhz6qvsgz0jvEkuoX4C+1uTXh9Sss7RsQteUvb0Sn9BtL+/iXRP1YXU8OyYeWRpmoGPmyLtIvoVlqWlE/C1MogPF9RXFLfSKFYJ8zUyw/CvqUNjVaDNxx/Kvkc0gz9MsFPJnSop8OVexJPpZVKhQjvwAjGaADoN+4sQAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,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>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAAAqUlEQVR4XmNgGAUXgfg7EP9Hwu9RVBABYBpJBiwMEI3n0SWIAWUMEM3e6BLEgE8MZDoZBCj272l0CWJABQNEsw+6BBCIAPEjdEFkgM+/YkBsjC6IDMj2LzMDROMFdAkguAvE/9AFkcEEBojmKDRxfwaIwb+AWAhNjuEaA8Sv74D4LRB/AOI/KCoggCzvgEAnEM9EFyQWwGy9hyJKJFjFAMkobOgSo4BeAAAM/ynzCFc9iQAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAaCAYAAABRqrc5AAAA3UlEQVR4XmNgGHFAAohl0AWJBQuB+D8UF6HJkQQ0GSCGsKBLkAJWMkAMoQiADPiKLkgM6AHiJigbZEgNkhxBUAnEv6BsVQZEoLLDVRAAqQwQDRxIYpegYkQDkOLnWMS+o4mBwEkg5kMX9GCAaEhHEweJNaCJgUArugAILGPAdLYKVAzZe3jBFAZMQ5YgiS2F0vJAvA+I50L5KICbAdWQYCgfJgajbwOxEBD/hfIxgDMDQmM2VOwflA/SCAOgQI1A4pMF0L1NMnBggHiJF4gFUKVIAz+AeDm64CgYzAAAyRwx8feOxB8AAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABlCAYAAADwBb/EAAAKRElEQVR4Xu3dd6hsRx0H8LEbSyyxEhV7i8bee8GOCRi7EkRRBHuJ/mHkoWIXNYkNRbFjR4k19m7UmKioqGjARsQ/NNbEer6ePblz5+3u2X333nf3vf18YNgzvzl7zu7eP+6PmTkzpQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAu+9GXTmlDQIAsBp+1JX/tkEAAFbDWV35VxsEAGA13Kr0PWsXbhsAAFgNSdYMhQIArKhLFAkbAMBKu2ZZLmG7ZBvYJhdqAxP5XN/vyt/bhoPcBdoAALC+rlfGE7bjuvLnMn7eVuS6x0+Jxf3K+jwQke98yOT4h135WtUGAKypG5TFE7GPlcXO2xcXbeoXLzt3r51yqTawpEO78r4mdqD9BgDADrhhWY2ErXWZsv/utV1e2pX3tMElnN6VhzSxA+03AAB2wBFl9xO24f7PnxKb99l+0ZUzS99LePXSDyHm3KwptxXnlv46h1Wxd1fHY9JbmPc/qm0YkfdkiZU2dp0mBgCsmVVI2OLjZXPCNtbDlrl3T5gc57zXVm3z3jfm95PXXOO9VXxfrnlU6d936bZhhpx7kymx+zYxAGDNZO/QVUjY3lSWS9i+PXl9aFd+XTeU+e+Lt7SByncmr3XylCc2x645z0e78reuXKVtaMxK2PIdAYA1dmRZjYTthLJcwjbIOXlAYfDpSWyWC5a+/XxtQ+ViZfM1Tm7q++LlZfwaab/dlNjhTQwAWDPp0VmFhC1DmnXCdtmy2L3ac1JP0rYV+SztEOswVLqsPNSROXEZwh2TpVM8dAAA7OVmZTUStpPK5oTt8mX8Xg8um8+p62PvnScPL9y+qudaeRjgT1VsnvTe5T3PahtGZNeJH1X1LBOyle8BADvqZWXzP8yDVZ7+m7XC//5y8zKesB3dlS+WjfMyz+tz9Qlb9MXSXzeL436+9HPGzp7ETu3K8847c7N8jvpzD9/lql25UxVfVp46/XlXrjx5zTWf2ZV31ifNkCdXszzHvsq97jE5PqfMn28HANsuvRPzkoJB/tF+qg1uQeZobaf0/MySf+7/6crX24Y5FvlNdtItynjCtqquUfqh01om9m8lWRvcuvSJ6uD+1fE0Fynb9xu+pvRz+i7XNgDAThtW1B+zr3OFpsk8pNxzu3qDFklsHlSWS9hi7Jo76Y6lv/8f2gYAYP1ky52xxCTDoFdqg1uUZSu201ivR/a9XDZhO7MN7Ecnlv7v8pS2AQBYP0kK/toGG/9sAwegLHK6bML2gK7csw3uJ0Ov4bxlLgCAg1gSgSQvmTyd47G5RdN64B7Xle+Wfk5Pesu+WfrznlSfNMOQjGQiewyT1FMyofx7k+P26b9Mfn96V35QNj7TMPm8/Yx3nsS+XPr3PaLsnbBl4dRPln5yfc6Z5t9tYD/JZ9/uXkgA4ABRJzbpdWoTndawvETrL5PXtA0PI2Sh1GnnTlMnbDEsF5GJ3YP6Wncp/V6Qg99Ux1naoT734ZN63Tv1y7I5YftS2fyefId2NfsY+z657iLlI8MbFvDbsnn7JQBgjST5+F1Vv25X/l7VI3PBDq3qjyl7Jy1Z9iLJWdanatva+iw5r07Yhtis+rBga3rfrlbFo10bLMfZUqn2+LI5Ycs52fPyilWZ1pvWfqaddEzp75ckGQBYQ/cpG0nK4N1d2VPV4wpN/dgyO2l5SVfeUNWHnq1FLJuwRVacTyzlj1W8XX0/x8+t6pGE7RtVPeccV/o5akMZ1tmqtZ9hJ2WYOfe7ehMHANbEe8reyUfq9TDjNLcte79vkIcR6qdHf1Vmn9taNmHLelqDF5S+bc+k3u5vmeO3VvVIwpZ5doOckwRpTPuZWtmLcpHytOENIzKHL/c8f9sAABz8krDUyUc9fy2T+5MgJIH5yXlnbJiVtNTxO0zq9XBq6heu6rVlE7Y91XF8qCuvnhxnGLc+N/Pb2mt9ovQPSQyyvll7zteaeuzWE7L5bBl6BgDWTJKPPGWZhWvfX/qk4FpdeUfZ2GaoTWJiWuzepY+fVjYeNsjWQ4OLTWJPrmKDM0rflpLE8Y1dOWtST9tdSr9vZOqZgP+K0idsqQ9Dl8NnyuK7w3ZJX53EIk/A/qz0w4tf6coXJufkidRBFgNO4nfLrnygig+yZ+Sb2+B+Mvw+AMAayoMGF6zqd6uOs+tBPTdskKUzWt8qfUJxSOkny8/yrjawj643eb1xV65dN8yRnrcM6cYNS7+t0WEbzf+XJU2yOXn2yWztKbu3DtqPS//75nMDAJxn6NG59KZoL/PTajm3nSd2sNnNHq4kkrl/eh5XUXa/eHvp5w8CAPtRhkmzbMY0WVj2kpPjDGEmmcjrvB6oegmRA81nS/9k7W7JMO2iw6L5nfP3yTD0dstcv7ZX8qjSDxU/tSz2+Q4GGfb/TFf+1pXHNm0AsFIyvLkOq+8fXvqh3t108zKesD26bG7PYsYnV/XtkASlnpuYoeP6ntu9x+yqqtfou3+Z/3cBANbEsMPEvMQgbR+v6g+bxHZSngLe6XvsbxnaneeJXXl2EzvYfgMAYB9km6xFEraXVfX0fia2k8uBtGveHchOLf1w8jDUP0u+b55mbmMAwJo7ssxP2G5a+rZnVLFhi67nVLGt2FP66z2/imX5l+G+9b1rvy79sHKWUxmeOs7TyLO+yyKyjMyrSn+NLJw8WPaax5f+PQ9sG+bI+RmibmO3aWIAwJoZestmJSR3L33bU6rY0Pu1nWvHva1sTtjGetiSECVxTMKW8+olZOa9b8zw3rzumRIfc42u/KP0D3MsK/dIAt3GHtnEAIA1c0SZn7BlPb201VteDTs+vLCKbdXrynIJ25Mmr6/vyouq+PB9Zklil/ZZW3LddfJaX+NWTX2WD3flpDa4hNwjPZptbNreswDAGhlL2KJNzoYk7l5VbKuyK8YyCdugPSfLjqTMM9YzmOVEsgPGIMOt7X1myXkvboMLmpacJVbvbwsArKHscLBIwvbBqn70JLadTiibE7bLlsXu0Z6T+rFduWPpt/zaF9/oyoOqeq6Z3rPMj6uHXme5flfOnbwu431lo+dw0H4/AGANZZuwsYQtyU/dnuMkWNvpvV15ZVXPFmHzPlOcWDaf86yq3u6YsYw3lI05e8PyIpmbVq+Rtqgs01HvPTum/j7Z23bsNwAA1sCQGI0lBsd15aelfyJzu7cKy0K8vy390hfnlL4XK8dJurJlVpK5abIP6ulNLD1bY99lEcNvkkWcM1yb47FlOWbJIsCnlf73G5PvmsTwjNI/vDBvhw8AYE1kLbVFEjYAAHbJMOQnYQMAWGESNgCAFXdM6RO2C7UNAACsjjO7cnYbBABgtaSXbZklKAAA2AXZGumUNggAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHHD+B7LSZAzFSax/AAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABlCAYAAADwBb/EAAAKpUlEQVR4Xu3dV6gtVxkH8GUvsUSNXeRaoybGLvbeICoYg534YoslghLLi13svYsYsFcMioIPGjuWgCbRqPiQazchigWJLer8M3s466w7Z5d72j5n/36w2Ht9s/bM7H0ezseaVUoBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJbTsV35XxsEAGA5nFv6ZO2J7QEAAHbf+V35TxsEAGA53KX0PWuXbw8AALAckqwZtwYAsKSuUiRsAABL7aZlsYTtSm1gixzZBjqX6soFXTmv9GPsVskN2wAAsLqOLrMTtsuV9cc/0pXrV/XNOqr05793FUuyltjVuvKyrvytOraf1b/zRV15bVUHAFbUrcvshO2bXflNE5vW/nAkaasdX7b+GtspPY+XbYMLemBXTq7q1yx76zcAALbJbcrshC3H3j0S206PKtt/ja328bK5HrF/deU6TSy/wZOaGACwYo4p0xO2A6U/dmoTT+ypTexwDde/z0gs5ZwqXssjw7925apduX/p15FL+x/UjRZ06bJ23TyWHfy8K++r6tNctyt/Keu/zzzG/gaJ/bYNAgCrZVbCdvfSH3tuE0/sdU1sM/5c1ic4s3rYnlb68Xf3Kn2i9uhJ/GZl+udm+e/kNed4ehVP/bFVfR7PKf3n5n1UOnbf0/42AMCKGPYO3SgpuEfZOGF7TxPbjPRgLZKwDbsy5B6+VMWHR7zTfKoNVN44ea0TrTtP6ofrh135XVeOaA80xq4x7W8DAKyI48r0pGCYrfniJp7YVu45enZZLGEbtG3yKLKN1W5Vph+P3Mfvq/qsc87jE135RxtsjF0jsV+0QQBgtdyuTE/YIsfGJh1s5dIeZ5X1CdsJZfo9Ddo2Y8nlonIvSRgHOeenq/oiHtqVP5Z+xucsuc7YpIOt7MkEAPagO5TZCVuW9cgMxtow1mur/LisT9hOLNPvKd5VDm0z1DezkX3OUY87S/0ypX88PI8rd+VXXXlMe2CGLOtRT2zIunTt9wOAbXfLrnyxDa6wttdqN9yxzE7YIseHWZNJRrZyx4Nvlf78mfGZR4e5p39PYr/uyiPWmq5zcVf+3sTymfQa3qCJLyLLaHy19LtAZMHenPMuZb4kNY9Pv9AGF5Br3aR6f1J1DAAOyxVLP9Yn/1gyeH2aLJewmV6PZZAV/6fJ8gv5LR7SHpgiyc9uulOZL2GLz3XlQ6WfibkM7lYO/Zvctiu3b2KH435deUBVT4/fNJmYkGVGNitJ8Qe7ckbZ2qQYAOb6Zz9Pm2U3T2KTNcMWSdjStu0l2klZFiPf6cL2AACwfwxbG83ytjawBK7eBmZIj2J6Cqf5flksYYv8fru12fc7Sn/9U9oDAMD+kTWtZiVsGRw+7+KhO+nMNrAFvlcWT9hyHxnHtRuGXsN6VX8AYI+7dukHhyfJyOzB/LP/yroWhxpL6LL+V5Y7yHmuUfq1ptLuI3WjKW5U+vavKocumzAkIcN1h9mEQz2D2Os27f0dLP22S6eVtWMbtc2SC4l9oys/K+MJW+IXlP47/qk5Fjcuh553p+S6s8YfAgB7TP7BD8nRdyf1WXsmjiUjiWUWXl4PNvFZ8vgw7dqlFy7f1OtzZW2t9txtPc5o6nWbzHCt6xnw3p4j9TphS4J2flXPWLGxweTteVpJ9s6bo7xo+MAcsvp+NiwHAPaR7PdY76/4sTKeaDy/qbdtrlD6LYSGhK/W1sekTbb7qQ0bfw/yvq5ndmN77rYeryl9/PTSr4NVa9f9yvt28dgkQXXCljaZuZgNwVMeXPqeydbYvWynXC9rsAEA+0ybVKQ+tt1O24PUfm6Q+E+repKfjdrW0ubbTSzJZJtM1fV5E7YYPpuSxG3w9klsMPb5JGxZ3X6QNqk/qCpZZ6w1dq7tlOsdaIMAwN6WRW/bpCL1lzexMe3nBok/tqpn/NpGbWtp88uRWJtM1fWx3sC6/pLJa701UJKrtDlyUs9M1/YaGX9WS8L2sKqeNjev6htp7631iq68fo6y0SKzrSE5njXrFQDYY+qkIr1oQ33YuzHjrB43eV/bKBmp40dN6vWip6k/oaoPMtGgPWfq9ePaWY9Ih9gwO/KFVayW+rD8xzsn9UG+d91DGFmhv06asjp+Vsuvvbqpz9uzuB1y3Vu0QQBg78oMyIw9e0rpZ2bmn/2B0j8WzUSE7LeYHqZWthdqXaX0n8+jzCMm7++7rkUfy3XGfLb0yVG2IfpAV966/vAlK9Xn8+kdO74rb5nU61maGWv3k648voqlTSYJpEcxWwRlRmx8bXIsJdsWDVJ/c+m/QxKz4dHs16s22YfzD6VPRn9UxQdvKrubsO3WtQGAbZINqg9M3udx2glrhy4x9s8/ezNm4H3tdV15f+mTvGmbZI+dr5Zkqx34X0uPX5LJu5Z+wP+wT+MgSWKSvsHwiDbbEmX/yHkc05VHTt6nRzDfNQlcLUuXJHEc888y3pO4E9JDmN84iTgAsAK+1pUnlrXxYLU28WrrY9LLlfPtZ/csfQ/cbslM2KFXcRmlJzZ7a7Z7hgIAhyn//POIcUweI548ef/K0icJzyz9Uhcbyfiv/e7isru7QGTT8kUei25XYjc22/iNpU9ms+n8vPe312WCSoYe5Ptmn1cA2HF5LPnhNrjCkrjutiwxMithy6PrL5d+N4dp7TYjiWvdi3b3sv5a7SPm/SgTYA5W9TeUfvIMALDisnjurIRt8PkyX7utkDGBO3WtnZBkrJ6sMuaT5dBxk/kN6qVmAIAVlAkXy5iwPars3LW2269KPzt41pp3+b5ji05/tIkBACvmuLL7CdtrS3/eek/a0yax53XlpCo+yPZm2ds1k1Kyrl1m2mY9uZeWzd1jzvXk0p8jM4UHqWcdvnnlXvKYt15XcJax+07sojYIAKyWY8vuJ2xxXlmfsM3qYfvK5DUJW93uqk19Ub+fvOYc95u8z9Izqd9gUp8ms37TNuv+LWrsvuf92wAA+1jWkJs3KdjOhC2zixdJ2E6dvJ5b+i3EBs8q0z+XNfKmHR/W6avb5FrTPhMZo3ZmV17QHljA2DUSW4XZ0gDAFMuSsJ1VFkvYBm2b1M9pYq3T20AjvWP148+cMztVzDL0xD2jPTCnfDbnaGPfaWIAwIrJDgfLkLCdXdYnbNlBY9a1rlcObZP6gbJ+X9lFpUcre9sOcs5Tqvos2VEjn7lWe2CGPI7NI+pazpPzAQAr7NZl/oTtW6VvlwH/W+1gVx5e1ZNw5VrTZlamJ62976GeJTIOVx6zZjJG3KqsnfO0yesi3tGVj7XBKep1175f1vayBQBW2NFldsL27K5c2JXfdOXXXflt2dqZi3/uyu9K38N0sPRj0vI+10p8o0eMSW6SENVmfZd5Ded5UVlLVN+7rsX8rlb673FGe2BEHg3/vSufKf3vDABwyfITW5XkMF0mJgAALCy9PxI2AIAlJ2EDAFhyJ5Y+Yas3XwcAYMkcLGYkAgAsvfSyZUYkAABLLJuVG88GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHvL/wGneoiuhtYTWAAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAaCAYAAACD+r1hAAAAkElEQVR4XmNgGAVDGtwD4gdA/BzKPw7EV4D4PxC/hIrBgToQp0PZIAV/keRgYnLIAqehdDgDRFIYSQ4EQGKaaGJgcI0BIokM8rGIwQFI4jea2E6oOFYAkujAIgbShAEkGCCSPEhiYlAxXij/MpIcw2QGTKtLkMS2ATErkhzDdSC+gCwABb8YIJp00CVGwcACAG1zIgSwOSMAAAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAbCAYAAABFuB6DAAAAm0lEQVR4XmNgGAUDAr4C8VsgPgPEgkD8H4gvQWkWmCI3IHYGYiWoxAOYBBAcBeJ/MM4XKN3FAFGIDLqxiDH8xiL4EIsYWOA6FjGsCsOxiD1DFgB5CF1nNFSMA1nwOFTQB8pngvLD4CqgACR4C4gvQ9mgkJBCUQEFIEmQVXiBHwOm+7CCDwwQhVlALI4mhwJcGCC+DgBiRjS54QkAahspjFGixIQAAAAASUVORK5CYII=>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAZCAYAAAAFbs/PAAAAhUlEQVR4XmNgGAWDDfQAcTW6IBBooQuAwF8g5gfi/0DchCQOUgwSQwGJQKwNZYMkGxBSDJuhYiggHkp7M2BKgvgP0MTg4AsDdg1paGJwAJKciMQ3hIrhBCBJdST+RqgYCKxBEocDkGQelM0H5cM0gEIRA9gyIBQtgYqBFIL4vDBFo2BwAACdVB8lFE+yYwAAAABJRU5ErkJggg==>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAaCAYAAAA0R0VGAAABzElEQVR4Xu2VyytFURjFPyV5hIyUkhibe5SBAfkTzCQpSiZKSAbyKDMzSiJjxuYMTAwYKSETTMzIs/hWe598d3X2Pnfg3u7g/mp1u2t9+3H23mcfkTKlwyAbKQyzUQwmVHNsKlWqTvO/Q3Vh/kd5VP0YfaluTD5P+bvJElpVD2xKbt+WTdU2eUEaxXVwyoGBB7Agq2bTg+yTTYn3l8O6uOI+DgzPbHh6VR9sGtDvGpvKgeS5vXiy2JOMqabY9OAYpJ010C6u3zoO5G+3Mkk7F5Z7VQWbHrSrYdOzJ/F+kTWwaakVV3TMgSE0QL2EMxA6bwnI8cIFWRZX1MWB4YkNT79kT26FTQPyfTYtOMyxAUZUk2x6RiXctk1chp0J8aY6Y9OCDkIDgDs2DJh4qO2uhLOEV9U5mxZ0cMWmIXZNdEt4AvBjbQFqjti0oOCWTQ8m3cKmoUnik0vuN3wp0kDNApuWVUkf4FI1wGYKaIvvJwMfK4ut76EsATV4wCiL4gq//S9e/9DdxaB+hk1lR1x2woEHF3Paovwrs6oXNvNgS3XIZiHAClSymUHBVy1hSHXNZoQlcWe9aGyoxtlMoVniV1fBwJuZxTQbZcqUAr+u9XI5+xUpEQAAAABJRU5ErkJggg==>

[image29]: <data:image/png;base64,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>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAaCAYAAADMp76xAAABNklEQVR4Xu2VMUsDQRCFR1NZBiRNQKzSprG1kMRaUgWson8ihfoDrNIoKBL8G/4Ba4OpBEGbNAZLwYig89g7bnyYc6vbRfaDRzbvzV3m2LmNSCLxv9hUPbEZiBPVFZvMq+qLzUCgjymbiSo5VQ3YDMC+6lq1xoHlPfvE3OzYoGLwwm+rtqTkXbpUrWZrFO2arIwPcfXQPWWWhqrG5i8cqHrZek9KGsbxAUZSUkTMxR2BOfgxXJs/uOWcjSUcmfWtePSCggmbS8gf0lIXdw9sZ86G6sF89wX3eWbTsi6uCNv3Fy3VCpuGmRSjsqDMF1x7yKblTootGKqaJquarniOw022frFBALzmtyOu6JODAKCPRzZjBg332YyJCylG4MysowUNvknx79b+GccHjtSx6piDRCIRAd8+9D1t7Pj0wQAAAABJRU5ErkJggg==>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAaCAYAAACtv5zzAAAA9klEQVR4XmNgGAWjYBSgA2YgrgTifHQJaoB2IP4FZXND2f8R0pSBaAaIYRxIYmehYlQBIIOeYxH7giYGAguA2BZdEAjWArE+uiAIhDBADEtHEweJgeIDHVSjC0BBM7oADOxgwAwKOagYK5o4WWAKA6YFS5DElkJpdiA+D8Q3oXwYCGCAqAfJYQWgFINsgRuUDxOD0bD4QFbLAsTeWMQxgDMDwtBsqNg/KF8IpggIOoF4BhIfBvSA+BO6IDkAlyvXAXE5uiCpAJRHQL4CATtkCQaIxaDgohj8YMAembh8RhVgDMSv0QWpAUCu9oTSNAOG6AKjgGQAADAbNNjENC/GAAAAAElFTkSuQmCC>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAbCAYAAACjkdXHAAAAwElEQVR4XmNgGPaAC12AWHAbiD+jCxIDpIH4PxSTDGAaSdacAcT5DGRqhmkgWfN+IJaFsl8xkKAZFC13kPiHGSCaGZHEcILvaPx5DBDNWmjiGMANiL8C8XkgPgXEp4H4BQNEcziSOqzgBhB/QsPfGCCa85DUYQApID6BLggEegwQzXPQJZABrhBlY4DIgbyAATiA+DkDxHnYgAgDRPNvdIlpQPwBiN8yQDR/QZVm+MuAkAf5/w8Qm6OoGAWjABcAAFndNJYyVs7BAAAAAElFTkSuQmCC>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACEAAAAaCAYAAAA5WTUBAAABX0lEQVR4Xu2WPS8FQRSGDwV6jUJQKIneR0WCREgkCskVCj8BhahUIiohUYlolKKSKDQanT/gMxoJSqXwnsy5a+67a/auG0Nxn+TJznlnZmd27+7mitT55yzAD3gOJyu74jADp639Im4z0dmBQxzGplv+6OoZ3cQVh7E5EbeREe6IwaC4xfvsGP1n6ZLKRct3Iyq64IZXD1jWZvUifIezyYgwW1LwInokPaE9I+M6j0LjtyU9YSUj4zrEOLzlMMSypBd4hEdePQZvvDqPSzjHYR66iRZrt1rtoyctURaC51dFI3wSN/mV+hQ+aQecp8yHxyt7sInDImSd9DtG4TWHtdIP7+Gw1ZtwLel1TMEHa5+J+ztwkfSKLMFnr/4Rb/DU2s3ivhk++l0pL9IJ78TdEaXBjkXuZlXsw1XKDqhmjjmoBb2yQ3EPs8861T4TsJfDmOirXejD9RvsytdzUSfIJzzgRM+mcTtVAAAAAElFTkSuQmCC>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAAAs0lEQVR4XmNgwA0mAvFjIP4PxDPQ5IgGv4G4H12QFACynQ9dkFjAxQAxgGzQCsR/0AVJASD/d6MLkgJAzudHFyQWcDNQ4P/vQPwFikHeGEkAFGjE4IEBGegCpIAaBgqd/oyBAgMuArECA5kGCAOxHZRNlgEfkNgkGxAFxLuAeD0Qb2Agw4DtaHyQATJoYjjBEXQBBogB3uiC6IAdiL8C8Tcgfo4k/hOIPwPxDyCeiSQ+SAAAZ/QwOnlwZbkAAAAASUVORK5CYII=>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAZCAYAAADnstS2AAAAZklEQVR4XmNgGPrgIhB/B+L/SPg9igosAKaQIGBhgCg8jy6BDZQxQBR7o0tgA58YiHQCCJDs3tPoEthABQNEsQ+6BDZAsnv/ogtiA2YMEMWT0CWQQR0Qn2BAhMJbIN6PomIUDF4AAIKgHWkcvvnjAAAAAElFTkSuQmCC>

[image36]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAACWElEQVR4Xu2WTYiPURTGH2liYaHZUFISTVODsmWDnYWUZGZB46vsSFYsbBUbRGHLgpUiNRtZ2ipqEjML8i2yMpOv88y5d+a8Z877/i9WdH/19H/vc8499573/34BlUrlf2WD6KPop+ihaGkz3MkW0Th07g0XI+dFk6Kjon2iEdGwaE9SxBrRJ2jNrC6OYy7vm+h9MzyfI6ILZnwdOnmj8drgYj/MmLX8BieS16YutoouQvN2uJiltN4sUXLkRTBnMPDOuDE3PyRaK1qdNC1abPIiOG8vtAbzI1jjGMr3PMNLzE8uKbATcc4Umr69EjJnRQe9GZCbvoV4LTKWfkv23MpJ6OTtPuC4j3iRScR+pl/03Jst5KYXQmteboZneJt+/7jp/O/x4dOLz4gXeYLYz3TFPLlpEjW1SbQuHUfxnpyDXkbfRdtcLKJtkUeIfbIf+mArhU3ziU8OQeuunAvjizlu208RK6CT7/qA4xXiRR4j9gn9UW92wKZHzZjzn5nxTXP8V02TkgJt9zTv18jnu5f+Eh/ogE0fMOP8+iOnofd6pmTPs/Byvua8XGCz8y2nEC/in96Zq4j9Lti0fcqvgtbgbeJrFTe9C3Fy9uyZ3C1aZsaEOXwae++O8whfXX6dXrDpw85jDdbyr7yoj1aYuMiM1yfvnvEWJM8XfQN9RWWWQ3P6jJeJ5vfiKfRz1J78K4jr/FZ9/lM8c9QH6MRLjQzltuiEN6Fz+K7kSeJcfm1F8DYq3dSA6B30w+lFOua/nnlgjvmlxpPPPOq16KuJVyqVSqVS+Uf4BdTYvbkb0mGpAAAAAElFTkSuQmCC>

[image37]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAACE0lEQVR4Xu2WO0hcQRSGj4IiPjrTGlAUREWbFBaCRRrRzsrO0ge+0MZKBUELH02IhY0QQhAbDQR8oY02FiFiIYKC+AIRRCtBQ/T8O3Pdc8+Ou/e6RVi4H/zsnv+c2blnmJm7RBEREWEZYnVqMyRFrBPWM+u3yoEzVgurgPWB1cF69FUQrbC+sspYWaxa1iqrThZpfpD5IUwMdfnToWgn8xs5Nh5h3b9mDd48Urm+CqIdkfP03VeRgnQawepifJ7wvIeQIJ5hzbJqVM5jmzXA+sYaVLlApNOI66HzVQx0jYtNMlvq3aTbyIH9Xk/mrLgI0sgG/adGKim+j/fJHOQv1tPAu2Ptso4o8QyBNVYfmbO7QGYMLoXAYEC3NgPQRu6t9Y/1oDxdc+vwfrJGRYxzh5rPwksKinu0GYBmMmOvlL9l/WRgpVHToBMK10K9CQp7tRmAj2TG4paRYGXhf1K+pJFMzZzwvOtbErqRfm0GBGMXlffL+hU2xlbTD4OXI7xxG5fYGGMloRvB/e2iiVWlTQHGHivvj/U98P1SxGDS+mgAlNpYnwd4f5XnpJhM8ZROkLkKU61INSXmEY+JuJW1LGKAmkOHly3iYevJl20CS6wb1gXr3H5ek/v/z6nyNLgoMKH3X2vCn44xTSaHOfA570/HQBOY/4niC1joq0iTPW1kKnrrZCTrZM5RxlOujYiIiMzgBYgAk8/yqiwzAAAAAElFTkSuQmCC>

[image38]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAABwElEQVR4Xu2WyyuEURjGXxFZWLF0SVEuEVb+BbK0tkdE2dmxsWCrlJ0FJcVKKUqsKAv+AWLhlp1yC+/TOWfmzDPHfN/MqGnq+9XTnPO855w5z3cXSUhIyJc51TibBdKi+mHTsiKm9qFqoxqzoepkM8SmmAWxMDSRWS4Ytx4Db436I14fnKq+bQ3qyixH819BDlSvkh3kMuCNBjzHgJQwSKNqV/Us2RsMnaU66w2SD0oaxG00bhAAb4tNKWGQHVWTbccN0mq9K/JBUUEm2YxJverI64eCHAe8VevhnmJckG4uRIFJU2zGhDcYCgLgLdl2hZgnFLzD1Ig0LkgPF6LApGk2Y4DHaQd5fwUBeI/gDCyqmsWMW8gYYXBBerkQBSbNsBmDfdUJCWtBaK+nh2YxJGZcJRckHaSPC1Fg0iybFvxhPteqC+IzZr0az3tSvXt9Hxeknwu5aBAzaZkLYq7l0MZyERo/b70q28eBQb86NSKTYTF1fvMH2RZzVO5Ut/b3Qcxni8+e6pq8EBeqezFrQWife/U31afqRfWlqvVqjkcxe/D3BO/GH1QMZ2yUK3yplCX4EMR9VPa0s5GQkFAe/AKsmIpvG/SkpQAAAABJRU5ErkJggg==>

[image39]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAZCAYAAABKM8wfAAABqUlEQVR4Xu2Uu0oEQRBFC0VExEQ09YUgCMYiBib+geADTAwE3yiIGOkqRqKJYmYoGGjgpoaGYuIHGKngA0TByEC9RXfPdNd2L7PLBCJ94LDbt2pmanZnmigS+X+swlkZWqzBL/gDh0SN4fwczsBJOAHH4Zg2F04pHYKdc8sJn/DGWr/BXWvNmHP4vLb6ciM08BGpmk2rJ/uGA7AXdsMurezLjdDA5leScLauv9fAfatm4H+iUYZ5Uc3ATzK04Gd+QYYe+GaP4T3cIfUuZaKagX05U0fhmk0tuX1yXRZunJch2KLSk4zqTOaGOzglQw98PXmOolgH4QMXZajhXeLKWj+T6ucdxoccIkQzpTd+ANvccnn4oCUZWmzAD3io19xv34ThhLIPzPCebYZm391yGG5elmGABlL9PbJA5R8VSb1Yb5M6tiByL9y4IkPQTqrWb2UXOvNRycCblG6Nhlt4KbISWkhdZE8WwCCpWp9eN+l1Z9LhUsnABSrt5a1yWGQJZ/AVPpDaB/mTXyj5MpmX7FF/djhVF9OXhQIcgS+U3ui03RCJRCKRv8Ev7lB6sWtzzIgAAAAASUVORK5CYII=>

[image40]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAZCAYAAABU+vysAAABYUlEQVR4Xu2UTytGURDGH5EtKRsrKzZKdkpYWmAlCxtZ6MXGh5DP4BP4FuxsJOVPiZWFfyUL/xIpMfPOOe6ZcY5OuXbnV0/vmWfmzn267+0ChUI+XaQb0ifpnNSh21GeSDPW/AvtkBCeFkigocCzLEFmag3yYA1igPRhzYA7/EMQXrhivEHnx/DBaw9yD1m6HXh8s9Gg9syTFtw5N8gc6Zi0RRoj7el2RStkqReHmFITFfyCenKC7JIWg3qDdBvUP+iBDnOi2014Ab/InpwgPNNvvGSQCdKrO4+jCnP0PQFMklaDmskJ8gKZO4D8Rb/Cg5YLaN8HDckJwjxDP+1p3RZmEQ/CsD/szjtG+65/6uoU3cG5l/SOxP34oxVtIO0zfch7IrEdMa8JN9aMx/Wh8UJGINc1bMPAM+tB3em8JI+QgTP3u6nbCv7uXJMuSVeuTvFGWobsZPF8m5ooFAqFGvgC4ANjTQ4aPPUAAAAASUVORK5CYII=>

[image41]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAZCAYAAABnweOlAAACcElEQVR4Xu2WS6hPURTGFzFAkiEDiWviFUOKARMDpe7kUldSCpFkwMRIppIhA1GYUDK6AzPlkboDMykT7/ebIq/va+3dWdY95+x1b1Jq/+qr819r7fXf+zvn7LNFKpVK5e9zB7oF7YG2QlugzdBQkmcK9M0HE7+gS9AuaFjKvcgp6JPoWOrrn+kxvJSm9gs0Ai2D3pk49QF6BX1MvznnRRLENvJ6a+ruulwbfrzVbVPXxjXosnT3zlwRreGCPRtFc/t9ArwQzS33iTZYuAZaAg1AC5K6Jsc70JX7Ca2CFkusl4Wm8Clk7QmXyxyCpkrZlN0+AeZKc4OKjPoAuAkt9cFElymToWM+CN5AM3ywBZpC+iae4yVTdvoEmCX9vXtZDV30QUOXKW0chPb6YAfZFO5H7D/H5Mg0aF+6nogpN0Rz630iQmnBUVPyYx4lm0I4jnuY5aq5LplyGFoourFyE34N/YBWNKVxzib1ETXlPrTdB3uwpjyQsf/x1FyXTDkOrRN9KjZAR1L8aFMahwPn+6AjakqkxmJN4V3meB4RyFrRzTtTMqXt9ZkpmnvmE33skNhCIqack3KNx5pCOJ5fM8KzhmUiphDmqHk+0cU9iS0kYkr+8/HgTTkt2mMSdMHlSqZws24jz2vQJ7qILuRfmULYg6dWfu59vM+UtnMKGfe8ogM4yVJdtFdmumj9GRfnFyO/Qpl8wHvv4mSbaO6Ai88WPeYzt8nleuGA7z5o4HvNL8DDpMeiBzNuih72Yj7CedHNjz0fifbM8M7bRfApfSJN7XPoOrRSmhuRRUNpxGfRT/JJUUMrlUqlUqlU/kt+AzqN4ZAhDmg8AAAAAElFTkSuQmCC>

[image42]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAACXElEQVR4Xu2WO2hVQRCGR7SwV2OjECUgFmksBBGF2NjZWSoxaqHiE1EbSYpUQUEQBDtJYx6NpNPqip2FhVaK2Cj4AsUXiO/5M7s5/849e3OuFxRlP/g5OzO7Z2fO2bNnRQqFwv/OadUh71yEM6ofqp+qa2logWWq+2J9XqrWp+E2BlRvxfpHdeKUVP2+qV6n4Xauq75INehwGu7IQ9U2sj9JfYJfVUvIfqc6QHaOHarLYvfc5WJM04dTS7dF+4mOBxsPMjKn2kh2pEmCKHqPWF+8mDqWq05Iey6N+Z2i8XYjZ4PvIvlgT5EN8NabJBiLnpF8/5vh+seK9mDZ4h68lG8F32Py3VOdJztHLHqp2D2upOF5sEeAv1L0hNj4QR+QKiEI+8BYEs0TiwZ1RW2Var66eCMw6Ih3LkKf6pLqjuqZakUangffHRf+Jg1nQdF7QxsbH8aurcLynto9FX3UO7tgXOwe28m3Lviw5PtDG+KEc6DoYbIxjj+TaWr3VPQx7+yCuEHx5GivJBvcDv5Vzu9B0SNkP5Hq3qNi33rEz9sYDML234QtYv03Oz9PjqJyicB/zjsdKHo/2f1i4/aFK9NT0Se9M7BbtZrsB2L9W+QDfvJcIvCv8U4Hij7ofBiHEyA/jOjPzZUFSxCDLviA1C/bTWInLWZSrM9O8uHbxW+Lwenqg/PV8UjsOMrL+KrUF+fz68is2FkVO+/TcMW/z5+AboidzRm8fUz0UfU9tIeSHgaSRwy/K1zvpuE2NqheSZUT2njrkRa1kecLsX7Qc9VnihcKhUKhUPhH+AU7fbhBvFLX+wAAAABJRU5ErkJggg==>

[image43]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAACEElEQVR4Xu2Wu0sdQRTGj0hC0BiJD+xipVG0EbHwHxDEv8DOSomPPCCdQrASxDLBzkZExEpBCwsbC0FRFIsQfKBoApI0QUggkJjzcWa9M+fOrLteCy/sDz525ztzdvbcmTs7RBkZGWl5z3qlzZRUsI5Z16xdFfOBftXaZF6yrkjiO6xSN5zPPOsPSQI06IZT0UfyjEem/YH18yaaD8b2FfKCtWm18eOgX6XlxVJIIbUk+U8sL/pxfFSxzshfiC9nmPVFmyEKKcT30mWqbYO+rebqK6RNeW9Yp8oLUmghB+a+k2Q5hJhhNVF8IdBH5T2z2rHctZBmktw51j6rnOQl9AyBp6xtcx8qpM74kf6yGpwet4CkIW0moJdyg9r8Y/32eBGhQgBm1S5myQ3Hg4QRbSaghyT3m/LXjR8xyWq32qFCxkg2AjBAuWKwJBOBzq+1mYB6ktxZ5S8bv4P1mPXZDXsLwfdCzyz4RX7fCzq+1WZCkLugvBXjN5Ks8Q2lQxPHNwNt8Il1bu41qQp5p01DN6tFmxbIPVLenvFD9FP+jGD8UE7Id6gh6TilA0wJSSzuQdEysUF7XHk2o5SbMRt4XcpbZU0rz2GR9Z11QTKluF6SHFtssGucKk+DjQIvEZ21JtzwDc9ZPyg35ldyjzI4HWDLxTNOzBVF3xtb2ihW9NIpStZI/kdFT6ojQkZGxsPhP6AVkwkLUTUXAAAAAElFTkSuQmCC>

[image44]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAAB+0lEQVR4Xu2WP0tcQRTFrwZtLIRgZSOINiqEdAqKhYUItqKCTTo1GhTEDyAWFtrqFxAstLEV/QJWKgiCKCFRQcRE04RE/HOP896+mcO8P+trXHg/OOzOuffO7N2Z2bciBQUF5TKnmmCzDD6qrlTPqnNVvRsusaL6r7pVdVMMoH5LNa4aU42qRlTDgbxsiJkUxdCkG85MrZgmQqrEzPfZ8sC9atEa/1UtWWMQfhaf9q28WPI0cseG0qF6tMZ9YtawwS6y96TqUrWpWlTNgTgvljyNoJaP5afADwl3noGHIwSqxRw95peqjs048jTyW0z9ruVhl3qsMeIP1jgE/hGbFvOqKTaTyNPIB4nOMYQmBp0M4/8hD8DHXfFRI/5dTAQFX9ksg0Zxmzl2w68ejggT5vs4U31hMw1MNs1mRvol+lZ7Jfpwh6WMaKcY+LjgPuIaTARF39jMiG/BH+L6eP/PGofAP2VTWRf/vKmgaIbNDAxJ/ILwO633vjx4a2xKfH4qKJplM2BA1c5mAB56cQva/iqNQfjgxKVm3tRIg5iiZQ5ItFjSpIgtkIfxAXnIs78QPKl9v2QgbU2HTdWN6kL1M3i9FvPwstlWfSePwd8PLHwSvOKMM01iYnti7tClG3ZAXlL8zWT6n1MJZN7m98yOmHtU8bSyUVBQUBm8ACsDkeS9cxGcAAAAAElFTkSuQmCC>

[image45]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAZCAYAAABKM8wfAAABuElEQVR4Xu2VzStGQRTGHyyUlGJh42MnK5YWtnaWdpKUkJQVZcWbrC1YKX+AjaKUsrCx9FE2/AHIZ5F8lcI5nZnemTPX3HdxF9L86umdc+a5d5733rn3AonE/6Sd9K2bGTyTBnQT0vuAnGOHVO1PFw8vlBd4AuLRgTdIo059BvHVOb1C2SO9Ij/wPbID6z/bbeprp1cYLaQt0gPigZ/Mb1bgT9KXU/dAfMdOrzBsyFjgYdKIGWcF1vAdY1+znlDwPl8nXZCWSDP+dMgmqdWMY4H5QbPkBZ6EeAb1hKIG/nq6Dmgi7Tv1b4FvSVVOHQu8DHlDvJA61ZxmCuF626r20OaswP2kadWLBbb0QXyzesKhEeJhrZDa/GmfNYRXICvwu6qZSgIzNkyMIZR9LPtgB+ySDpTsQTzmB4HRniPjOTc102B6Y6a22PO528mlVtWLEH9J9X+lkivSgfAKr5rem9Nj8s63QJpTvVPIG6Yi8hZgeiGecadXb3ou/NXj3rzqu5QQHncD2f9RTiBGfheyeHzoOYRH0hXEc2lqSxdkcf6A8L7nMb8FYpQgd+oO5Yult1UikUgk/gA/9ZaHM14QChEAAAAASUVORK5CYII=>

[image46]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAZCAYAAABU+vysAAABUUlEQVR4Xu2UsUoDQRCGp7CxCAhil5BYaWNtETBPYCvYW/gEYiUIVrYSQfAVBDshgoU2oqLgK6igIIKoIDb6Dzfrzc7NitEr94OP3PyzuZtcbo8ok/kbbfhpQ2EFflDR75le7fBFvEFe4YWqn+CmqmtlAN+oOkjfySacrBaacB8+UvUCqbvE2aoN/0u40LCD3NvQsAiv4SGcg2dxO2YPtuR42EG8PHAKl1S9DR9UHTEOj1TtDbLuZAuS2VzDvSmTJQexJ/IGYXjXHKuaT8jreDunCA/+FRV/UZIdOG2y1CDMGnyGW1LzOj2cxwuVd46dj9sFB/DEGL7Ax7vl0gqj5N96DW/xQIfKl+Gv8P73tmSzKuOtbtdZvL6XuXiDdCWbkboh9eT3Ch9es6HqMcl+5JKKd8KNyMfnqh8ezjv57Kheine4TOWPu4Uj0YpMJpOpgS/rxGuJHVwhBQAAAABJRU5ErkJggg==>

[image47]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAZCAYAAABnweOlAAAC2UlEQVR4Xu2XS+hNURTGP6+8khQDBiRRhEgGUgZMFKUYSfyV8hh4ZGKi9JcylDKhSIkyEiWkDIgRIgMGTLzf70fIY32tve7Zd519zrn3XwZq/+qrs9daZ597vrNfF8hkMpl/xwjRA9Ef0U2XS/FRtMIHhXWiL9B+zrtcikOiz9B66nt7usQrFLVfRedEM0TvozjF3/da9Cm0f4gmowvWQm8cFNq7RB9a2TIboPXelBuinnA9AMUPHNyqqOay6BS0vo7T0Bq+sGcpNLfNJ4SX0NxMn0gxBlo8JIrZy1RhXys2pb/op2h8FBuO4is1QVMGQuv3u5yxA/rhmkzZ5BPCODS/V4tU4TDXjuEwJd6UxSHm+0rFUtAUUldv8SZTOJI9I1HfdxssuhOu50HXlirWQKca8aaQw6KJLtbpDzFTNkLrx0Y5MlS0JVz3xZRr0Nwin/BMhRYeF92GDvcDIZaCi5eRMsXDEce6Rz6RwEwhvOde1CYXo+smU3aKJkEXVi7Cb0S/RLOK0mpWIv0lf4u+udgLUb+o3Ykp3B1831XEpjxE+b5n0XWTKftEC6GjgtN6d4jvKUqrWQItfuril0LcYJ0NXaPJlK3QGu5CnRCbwq/Me1eH9gLRtCLdaEpq+nBZYO65T3gmQAuPufiZEJ8b2n7UkDpT5kBHWzfEphD2b33wrBHTF1MIc1S8QyZh0UkXOxviU0L7itP1kL8b2jGjUTax2zWFHIE+g1P2hMs1mcLFOoWZstwnPCy672K3QrwKmpUaKZwq71yM1PVleFMI7+O6xDOQj9eZkjqnEDOlkekoF7Ld62Ix86E1613cHurVNJVslzrq4twx/L12wEuduHugue0uPgp6gGRumctVshl6g/332dueboMj4Ql0SjwObbIKZTNMF0JNCh4HuPhZf2+jHL98/BI8EnBTsFruiFdFs1F+Jg2lEfwfxi35INTQTCaTyWQymf+Sv6SF/EvPNLBzAAAAAElFTkSuQmCC>

[image48]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAACeElEQVR4Xu2WzctNURTGHyRiQkjIRxIjZSKFkT+AkJHI51RIpMjA1ECvKBEDUZgYKJkpE2MDCVEonyUGSL72c9da96yzzr73nPed0f7V6uz1rLX22evsc/e5QKFQ+F9Zm+xjsj/JHiQbXw8PZWWy15Da4yHmuQnJeZRsSohFliT7BMk3G8ZBVHk/k32oh5ucSXbO+V8hxYudNoirqC/oIpo3nAjJWaD+BPXn9DMGsw6yPuavDzFP14fTh4mrMlrbBGyCORuDTm2z8+9D3gTPKbTPT9j0NkjujxAzJifbj25r7jEV+eScFjkLyZkbdGqfg+/fJLJa9Tas6RsYnH9Xr13W3OdksjVB6zLBC0jOtKD7WnuVj1XhHgtV3xT0iDVt88SHR97ptcuah8Li31EMXIbk5Xbabr5CxzxoPLNUPxL0iDVNck1xs5brOBfvzENIcdsJOwP53fI356I53leFe0xX/ULQI6zfruM9kJr5VRhf3HjMTfNAYyF3ogu3Ub/RS/VNW6pjHjSemarzpzUMNr3D+ax55vzrbjympu3pT4qBFjYkewx5APy++5uP0/FR9Q3uFvWtQY+w6V3Of45q7hOQ37ox6qZtsZ4rwe8K57kW/HgA2end9q1m07udvwhSt1OvnlE3nTu0fgV/S7LZQeNN7jjfDi4P5+Y54TmMZl4ONr03aKzjnP5hmN5lzh786FtBNMNe0zgpff9GcDGnnU/4FzdXNxK0HE8gf0f9a3wezflIbn1Z5qHZqNk3l0duJTsUNFvAU70eqIf72M7y4OFDvlQPN1iW7D3kn9wrHXPXjXtuzPneQvJob5J9d/FCoVAoFAr/CH8BqhDIf7aiPZ4AAAAASUVORK5CYII=>

[image49]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAACf0lEQVR4Xu2WS8hOURSGXyIUEjIRSWYmogwYmEgGRgYMmBm45VbKiImMUAgTJCUSCRO5DZBcSlFiYCKXgRS5X3Jbr73P/6/znr33/0m/wdd56u2c9a6119n7O/uc7wAtLS3dzkbTSjUdm03vTJ9MyyT3Nxwy/TJ9NS2VXMUU022EusuS8xxEqHlqmii5LMdN3xAGUqvq6R4emi65+IHphos75YVpsotPI/TyzEGYS8U0iSt+mua5mDXzXdwRuUWPRPqi9EapWWC66YSaaPZmrDuON+aWi7eiOW5BwuuT3KLvId2MHrdXp+wyfVET9d7jYsyj52L0K3jO3afQn6Bmidyi6ecWnfJzzEKo/2EaEb0lptc9FcAWpHseRnPRqWed/h41S/T3oslj9I67GmPPmZhT9qG56HMurqB/Xs0SHLBaTeQXl/P74iN6x1KjXe5a9JTdCP5408B4zpegQl9/yCIcsEZN5BeX80uwfqppgOltjH2PYxJX7EXwB8WY59wVCv3rapbggLVqojmxipyf44lpm3jLEXpw+5LcM139t1fw/IKLK+gfULMEB6xX03iP9EToPVKzQKoHuWL6HM9nI9T9y9t7sZolOGCDmsYipCdMb4aLh5nWuVhh/Vg1jaOo//WxbqGLyQfU3/JcsM5pZsIrwslwwA5NRJhb4eLt0fMwpnjxFPtN38UbjmYfbltfx+efNZOcx5cfvaHO44686+IsJ02vTM9Nz+LxJcIXkId3kRe5Y7qP8JHByXh4d/iZeUR8z06EPrwmj+wzuFYR4DX4lj+FUDe3nv4DPebOmt6YbtbT/w9+dGxSs9sp3eWuZAzSn4ZdzRA1WlpaWvqT387exkIzniziAAAAAElFTkSuQmCC>

[image50]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAACjUlEQVR4Xu2WS8hNURTHFxKKEsrgK18xpBgZSUIxYSATkWLk/YjIQBlQFCMfJvJMCgOUvDMxIiXFxMxj4P0o5BHWv7X2d9f933P2PZeBus6v/t27/mutc/c++9yzt0hNTU23s1m1ks3ANdUv1TvVcsp1wj7VN9Ub1TTKgQuqg6oJqgGqyaorqimxyDkqNqbXqpGUK+W02ADQCK1qTveD3GD/vsjjl410ZT6odob4i2p3iMFtaYwn6VRThd0M+Es8HuJxb39FRcomfVl1nrxLYvXzyc8xS6wnMqrAu6XaqDqp2kS5BG4W9/UVeG0pm/QPsdzC4E1070Xw2pGeKCauGLghtpI50MPXmuceVr0yZZPuUZ0gb4ZY/T3yc6D+O5ti/oMQX5c/m/R091aQn6Vs0kVcFaufxIkMqP/IppiPxzWBa68XezKOeZ4nUjTpZe7tJz8LGlazWcAgsdq7nGgDet6yKa0TuKjaEeKhYvnZwcObnyf9yD28byqDhrVsFoBV6eSxTuD679kU83+ySfCNSTd+rsdjVDfd256KqoCGdWwSD1Vn2KwIrv+VTTH/cYjT1hjhSScOi22Da1RLxWpmNlW0AQ0b2AycU+0i7ynFOcoGDu+Qfx/nMT+iZb2RPdK+pgU0YH8sYou0ntbGqg6EeJjYC6gMnLJ4UOmQkVZ3vMfx/wvgYetMHHEvgvg+eVnwn0DTXk5I41BRpDmhLnlTg8cgjz0+cUda3+ioGRjibe7hhZbAthYnvZjiLGdVr1TPxB5VfOLAge0iwRONioNboHquOh48plesDy+dJ2L1DK6J38eenn5neFOFAf+T6rPY//qfMUK1lc1uJ7fKXclosXPzf0VHB/2ampqav+U3gujDUR0mUdgAAAAASUVORK5CYII=>

[image51]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAZCAYAAACB6CjhAAACkElEQVR4Xu2XOWgVURSGj/uGSkDBQmIEFURbwXRCChEttbCJYqooioIIaiGaIkKSJgZRsHNfEBQsElCfkMpCXMDGBUERRRAVxH05v+fc57ln5s28l5cU4c0HP5n/P/fOct7MnQlRQUFBQcxeVqcPa2Q26xnrD+ueqwX6SOrfWUtczdLCekMy9npcGj3Ok5wIDgJtj8s1sZVkH1PUH2J9LFcF1E86v8H4wAnWD+OvsgaNHxPqacB8kvnTTRaaGnjoPNiYkrWlZPAfXDbq1NMAf7FgpvNpY/DIIFttMvgnxoMFzo8J9TbgkW63klyYJ60BANkF3Z6j/pj6tfT/kcoDx7zIes7axzoTl/MZaQOWk8w9y3rAmsUa0MyS1oDFmoXmYRGG38+6QrKvEsk6lcVC1hfjmyh5rFwwYYcPq2AzpV/cb4pP6g4lxxzX7LN6LJBp+4JHcytxmfXUZX4BzgUH2enDKlhPMve1y29pboE/qtsTWMOa3dSsR/059YFfmlcCjwrq31hdJHdAzWAHu3xYBYtI5p52Od7dyFe5HN8B+MVxos0kY45obYv6DvWBT5pnEZoX9Dgu54NJu31YJZiLBchyQ/NlLresIxkzSf1c9f6DDA3LagDmBSayLpGMX2PyXDBhjw8VnOgKHxow1z+D9zUPtKufZrJ3JLetBWNOpWRZDbhN8asUYOE84LKKzCM5QK8vkDyreSewkpJ1+MPGH9Rssno0FH5qeYSwTXMLPD6QKlGieMEFmIMPtEyweuJXeMV6qX/fUvK1c431wmUeLKA4aPhfoDsu/+MrySfue9ZP1oy4XKaf4n1tissJSiRrDe6m8GNlNWxE3PVBo+Fvy4ZiiGSdaFiW+qCgoKBgPPMXD+a5+AF3LL8AAAAASUVORK5CYII=>

[image52]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAZCAYAAABKM8wfAAABtUlEQVR4Xu2UTStGQRTH/0goSbK24ANQvoJvoKzspLzkLRYs5CY7VhZWFkqysKKUt7JhY0WJD8DCy0aUhOIcM/e5556583jyIGl+9e+553/OzJx7n5kBAoH/xxipV5uCSdI96ZHUpXLMOmmB1EQqITWTtkgtsqhYVknPpDervnQ6xxlpV8SnpEMRMwdI5om1kqr4ZnwN18DkNOzVinifNEJaJo0K/8fwNXwMf8OLIt6D2Qq/hq/h+O/VaJ+3zFcaLoV58QvSDMxZKohiG94mDcGciSWb6xH5LMqQnkPHeeHCfm3CbSxG+xukSMSVMPk24Wl4PT033zYFwQMHtAm3sRifL/mspg5JzTypIZ3ODw8a1Cb8i2q/XDzH6JosOpHUse7SaT9cPKxN4gHZi7J3bp/5y3C8maQ/+KzhChVPw9RHys+EC/ke1XQge1H2Wu1zo431fmXvVXmSKdK48k5IO8pzqIeZfE4nLJyTJ37WehKO+YqKmbAeHz4fEdx5ruC+eI410i3pEuYe5N9rmKtJUgUz8RHMF3iCe+dyszzuBclWqE5VuESkdtINkjHdsiAQCAQCf4N3tviCpTRm82IAAAAASUVORK5CYII=>

[image53]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAZCAYAAABnweOlAAAC5klEQVR4Xu2XSciOURTH/6bMSbEgFBIpInsZNhZKyYr4FhbYmEoSUoYsxUJRpkJZycaQnbCTlUhszPM8z+fv3Pu95znvfZ/7efsWFvdX/97nnnPufZ577vgChUKh0P2sE50STQ7lSaLjorWdEcob0VLRUNEQ0SLR60qEslz0UfRbdN75UhwUfYDGU1+r7iaeoxH7SXRONAX6fdFOvRO9EL0P5W+iCegiO1FtjLpeiVB8DDWqEgFcE3WE515oxPXtjGjNJdFpaHwdZ6Ax7LBnPtTnB5Q8g/qmekeKbaJ9omOiLdDOpGCDu0X7RXOdj/QUfReNMbaB0HocpRxMSm9o/F7ni2wU9UE+Kau8QxiJxiBlYSJme2OCXGPzkH5pypaCSSF18dGeS8oK74Au+bq2K2xG9ySFHBKNdbaufkhMykpo/AjjI/1Fq8NzO0m5CvWlZnkTm0S7oBWOhN8DlQiF9tuiG9AX/IBO9zoGQOvd944EMSmEdW6ZMrlonnNJ4ewfD91YuQm/FP0UTWuE1rNedMHZ2PCOhM1umGeDrQ6eDrmYiE3KPTTXe2yec0nZI5oDnRVc1tuDnYdK27AB/1EeHt2M2eodgTVQf6uN22OTwlFmXV4ByEw0rgwkl5TU8hkM9T3xjhQ9vAE61XxSfOd42jDmprOTGaJf3pjBJoWw7dgG7xqWdpJC4mDbEzIJg7jmvM0m5U4o9zO2QcF22djIMNFnZ/vXPYUchrbPQTvpfLmkcLNOEfu10Ds8DNqQsNmkcI3zlmqJR/BiY+NsSt1ybVut8EkhrMd9ibPS2+uSkrqnEN+vlvClw015FrTiRGMbLbpryuQLmmdEfKlXbinFU+qos3MZ+7rxgvfW2UkH1MfDw8K/JrxA0rfA+VrC5WM7Ma7q/ssyqO9h+L1SdWNJsKfkTzfLCejmxyX2QPTK+DjythP8P/MIjdin0O+YjuZ3MqFMBGc4+8drRu4KUSgUCoVCofDf8gfK+fppzeHe9gAAAABJRU5ErkJggg==>

[image54]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAZCAYAAAB+Sg0DAAACIUlEQVR4Xu2WvWtVQRDFBzViIhEsooVCioDGT1SwsBMsBJUUIY1dSJfCxsJWLESw1n9ASRFJIYKIglZBBD9QxE4LyQcYjGKj4PccZ8c397ybfZckRYr9wSFvz5m9787N7r4rUigUVpNe1TvVH9ULypwHYvln1RhlzE3VHjZreKj6KnZd18dKhciPkEEXq3E7o2KFXWmMCV/+p0bMz6bxQiv+x7Tqd8qgvdU4ywexOZs4ULaIZRs5qKNP2i/kN+TcU90OY3BXrGaIfHBElt+QPzQHzfwiLwvfPOih8U+xmpHg7UseboRZrYa2qr6HcSNwkdfp8zGxvcTsUN0g77jY3Ofkg5U05Mtqm+pbK24GNi0uMqF6pdqsupa8TtwXq9vPgaysoW7VzvS5yX1U8M3NE7Gxc09nvdicpxwkvCEsy6Z4Q4dUn1RX0/hJLOrEabFJ8+Q/Sv5SoNm6peZ4Qwc4yOANzQUPhwG8weBl6RebgN+MyJ3kHyUfvFHdYpPwhg5ykMEb2hA8nLx1KygLiifJ8yN5F/lTqsvkzdAYeENYPk2pawj4nn5M/pKg+C15L5MfuaAaJ2+76jp5wBs6zEEGPuUi/l/azUEdOKX45jG+FMYnklenk6HOOSWWneEgA16nMGeAA+W8tL4Pp2BHzokV+7vclWrc1kTUulCHVyE86VmxpYi/8N6HGuaZVOfgUEBzzrBqMeR4z8MPfaFQKBQKa4q/wsGrwGmult4AAAAASUVORK5CYII=>

[image55]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABIklEQVR4Xu2TvUoDcRDEF0FiFLHyIXyEoOADWKQTLIQ8gZWIXQhY2dgklnaSOkW6YGGrGMTaNkQEISAI8XPH3bts5nLp0t0PBm5m9jbJP3ciBYtmXfWs+lU9UJdwohqLzexSl6EmNrjsvq4apa3xrroP/k11HvwUm2ILV0IGDyW0yIPkvpnwArBKftYMQHbKIUDx5NcVsbNl5i0dcrjlxbXqUbWmanoWmbc0kx/kFD+qj+Abkp3Z94xz2fNwQPmN5xH8+7fBv4jN4BGbouzFFeUdz7cpZzBzwSFA0aWs53kpZPxMViX7a1L6YmcY+VJ9B38otuA4ZJ+S/+b9gxsu/frI/dKkTo9pw/2d6nVS54Nvg08+48LBC9EWO5od6goKFsUfIspQgGRkn2sAAAAASUVORK5CYII=>

[image56]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABIklEQVR4Xu2SsUtCURTGDxU1ZkhTgWtzoJPQ0NQeEkE0tbaGQ7Orgjq4+U80hDQ0hA4NKS0tEc1R0Brl93Hv1XOP97W5vR/84J3vfO/xvE+RnGWyDu/hH7w1u8AXPINbcBMew8+oodgX9zAWSdnPFmbW3aih4PIpkU0SWQN24aHZRWyLK/dM/uhzjZ0zORdXbpr8zucaO2dSkvSbvvm8qDLOL/AZPsAfuKb2ESyPExnlR9PZhppvfJaEh84l/1bkStxHYrYSSgn2xHWu7SKLb4nfogbf1Rxg58OG5AAemYzlgZlpQWU7PmurbEa4IRDKqyrrw4qaSUv+OaJXeOmvw/lW5+sZv/DCX5+K653M14vwoUPYsQsDP+II1iX+JTk5y2IK3/xJ9g5bqJkAAAAASUVORK5CYII=>

[image57]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAZCAYAAABU+vysAAABYUlEQVR4Xu2UvytGYRTHv0kWBgOT5S2DDP4DGezKgDIqgx8RZUMZjGZ/gTIY/AGyGWSgxCIbi0xCScT3dB7vPc95n+t95Y7Ppz69z/M95957uj9eIJP5O2t03oeGLfpFP+lGXPo/+/QdegFxIS7XOaajZr9On8y+UsoGaaMPPoT29/mwCsoGGYfWPJLVfFgFZYN0onh0/SEbDPtmTNNLekRH6FlcTlM2iHCIYphz+hKXk5zSWbPfRfoRNyAXWfSh4QrFMOJQXG5AegZc1vIgSz4MvNLJsL5BMUxXvaMROUZ6LqCPqGXkoGUfkj164rJhaP+1yz3PiO/iWFxOI40rPoTmPy+pZRNaK6PXrGso/q+aIk2rPoTmEz6Evoi3PjSkLprKInqgTTu+QKaQPoFkHT40SH3b7LtDluSAPtJ7ehd+5c2W22iZgZ5EPtuPsJbhf+ONzkF7RTl3e9SRyWQyFfANtbhcoMbibjcAAAAASUVORK5CYII=>

[image58]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAZCAYAAABU+vysAAABPUlEQVR4Xu2UPUpDQRSFb+ECLGKpSAJa6wLcQbCwiMFOLPzBPwhYiylSBEsxK0jhCsQNiI3gFrQRS1FEBD3HGXgzl7lklFfOBx/J3Hvee5e8yYgUCn+nB3d00fMNz+Am3IDrsOOdDXL/Zgw/xT2I7sbtX+ak6qdkv1asQfbgOVyCC7AFm/AEjoJcbViDXOsCmILvulgX1iApvnTBoAsf4A1cgXdxO03uIBfwVBcT3MKtYM3rnoO1CQfhfpgEczkwt6hq2YPs66JiKPmDvInL3ot7RdnwogNdVDCT9Z49rxL/1dtxOw2DR7qoYGagiwYzwfd5qc6riTB0rIsBq+Iyh7phkHpoqhbREBfiHrC4FJfhMZ8Ds/1gPe1rSa7gC3yCj/6TO5s/o2ZN3I2WdcPgA25LtT94bx6EhUKhUCs/lQ5Piw32x4kAAAAASUVORK5CYII=>

[image59]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAZCAYAAACPQVaOAAACX0lEQVR4Xu2WTYhOYRTHT0I+V7JT86J8hKwkWShlQ0opNbspGxtbWYislC02lkRZKCmWM7L1bWZlzPieQT6jSML5zznn7dzzPs99b3dB6vnV6X3O//yf+8x55t7nXqJCofA/s5hjguM3x91QMw5x/CDxbAu1FO9IvBZLquUKc6nqfa/6JMd3p2P8keMDxzfVbqm3EUMkk+Zofozjc7cqfOW47XIsdtLldbyg+k0E1zmGSXwprNnIfMrXelhKYpzntDj5TMiBzWsC7pjXVO9/S+2aBY9IavtiIZK6yIKQpzwA2uEoJkCz60j8e0MNbOHYSO2bvUdSOxALEZhGdYxF8exGcgtBw3+sH2gWwP/TFxR7ZNo2W1frspbEdIHjAcdCjtOqeXIXy+kRa/Yipf2X9bdJsys11nDsVu2h82UZpPQf/IvkpDOOU68Hz0dqbgqcqGAWif+sqx3hmK3jJs1u19jBsZNjjORu6XSdGXaRXGAq6KlFcRrfdPkbEg9eRf146sbYSH9tP06ta9Rt7DWS2p5Y8AyQmM4H/arqm4J+lOT5OqU5PH4Dcjx34yGSeR2Sx8YfKm2bBf3qM8BwKWi2U6uC7rH32+pYSID3rAfznnDcCPpfafZx0O6rbgxovtlpV1RrwquQj5PMfRb0ts3uJ6nhy6qW9dR7EeQ4lIytqm3QHK8n5Mu7jjx2uPlPxWWqrXAauKP6oqCDXLPWaKqW5CCJ2b6NT1TLM9iBhP8SfjuVahrs9EuS23iaY8TV/K19juQLyrxY45PWcEZYMxY4ffGN/EX9aLhQKBQKhcI/4g8/5+BcGyHybwAAAABJRU5ErkJggg==>

[image60]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAAB/ElEQVR4Xu2WzysFURTHTxGWspVSlFIoW1bsLFDW5GcpFrKwsPEP2CBKtjaIKCk7G6X8A5JskN8UheTXOe/c2ztzZua+583iRfdT32bu+XHnfOfN9AbA4/H8VxpQ96hv1AGqNJgOUIh618EsaUOt6GAE1ahH4HmsXIxDuu4DdRtMhxlGzYr1MnBzo4gRRyaezRCSCdQzpPtWg2knLag54L52lZP8eq6o4qiY5Qnic5nIxXQ3cF/c01WCGgP3zCHOIVzs2iAfpqkn7pq75uiaOSOTwM30/kWRD9MFwL0LwXSKa3PM2XQncOOMTgjyYZqIMtWEqjPnUfmMTAMP9IlqVTlJUtNrOuiATPeY80Hg/op0OjWLJSfTlnLg5m2dMCQ1va6DDsh0r1hT/4lYy7+/RKYJ1wZJTW/ooAMy3S/Wp5C+9hTwu25xzRyCHuclFbMbNKs4kdT0pg46INMDYl0JvEefOUqyNt0F0cU2Ju+kxWWa/jNHdFBAfVs66IBMD6kY7fEFwZth43FzhaDCYrGuN7EdEZO8QPzm9sK1OoEUAef2dcLBMfDnqLz5ixB9/V+ZLgO+c6Q74Mb5QAVDn5KXqDOjC9QDqkrUdKAOxZoYBd6X6qmPPoauUK+ySFGDugGupR46p1/dsifO6UuN9rNz0YxvIu/xeDwej+eP8APLQa8gexlTlAAAAABJRU5ErkJggg==>

[image61]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAAB0UlEQVR4Xu2WSytFURTHl6QMPErKzEDJwIgZH0D5DMzlTZkxNaMMPCYYKUkpEwNlYKSUmclN6QgDKSV5Rqxl7XPv3uvctc/DxK3zq3/nrP9ae911uvs8AHJyctIygxqWZgbi+mygvlHvqEGRC+lAPQHXnaGq3XSUbdQH8ALSiJtOTNI+t6g2K95DnVsx0Yo6seJ64J6NlufFN0AatD7dqB1pAtf7YmIMVZCmhjZAWrQ+S6g3aUJ0cIq7hDeJCoSnog2QFq1PL3DuC3i7EAOoh2IFE27PZeE1WLEXbYC0+PpcQGnQYxNLWqBUE154u1MRAy0alWYG4vo8gztok5v+pQfcmn037YcWjEszA74+lOtEVaEeTUyymUNdmfMhKNVsFitioOIJaWZA6xOg5oUXDrpiYnpfyAsjXqC8XxYqnJJmBrQ+2iBHqFdzThd0beVstPURqHBamoZ+4C2RBK0P+c3SRLZQ6+ac1mkDa74D/QAVLsgE8H6mXJJGvj6rqE/h1UG0L8V9wjtArQnPYRd1j7oB/kvpeAf8uWFDT41AeDZJ+ywCD0q1dKQXZI1TAVAL/Mil/KU5zjoVf+RUGpWK3AIVySGUv1ErjlSfCDk5Of+HH1gCi6JozbfjAAAAAElFTkSuQmCC>

[image62]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB8AAAAZCAYAAADJ9/UkAAABaUlEQVR4Xu2UvSuGYRTGTxkwKCmTAZMJZTEgg00pUf4Gw7tYJANZxYBMVpPJIH8Ai8HmH1AIhVIGH/m4Tve5785znnO/b1aeX10993Wdj+d53573Jar4zwxCj9A3dAa1F8uJReidQt+4qXnMUOjNMg9tK79PYWBIZcwLdK78E7SuvAfvqXtzr8Fmu8YznU6muaPynhLXVG6wQ9ZHOFuyIZiCFig/l2WZwsCkynJLOONPaHmVa27OZZpC85bJc0u8/BJqkrNXd9mADqBPaMLU1qi8ZE4ynY9Bq8rbekO6KAwcmZzf9hPl7yn08U8v8qHOzK9vzuSGVqBnaEc898QHuoBa5BzJ7Unw17xnsjg0anJNK4WePvGnjuIePtekLzFL/tPFLL443eKHUwfRoWT18HYX4GKz8gOSHatsRLJ+8W3ie1OHT8Obd0BfogcKzfyPZokv2I1cewrVIm/QLXQl4tnNQkdFRcWf5weUSHlOoptYuQAAAABJRU5ErkJggg==>

[image63]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAZCAYAAABKM8wfAAABqUlEQVR4Xu2UPSiGURTH/z7KKoNFSTEjK2VgMonVqshqkIE3GS3EhCwyyEBJkcnEaLMjEpKvQb7Oce7Duee9V+hV0v3Vv7f7O89z73me+7wXSCT+Dw2UC8oLZZdS7pc9SikPVirqKHuQubZNrSD0UabUeBGyWJNyzIHzWUK0wq81mnFBCDUQchnXiNfY9xvHu8G7VjCOkN/ATxquhHj+1Ww5/2sMQxbosAVHrOERhP0Cwl5TTJmjHFLGKYN+OU4nZPJJW1DEGl5F2M8g7DNK4NftOMoEZZnyRGkzNU2s4R2EPT88+ypbcAwg/741M/4UnpgnWLcFR6zhJYT9NMTzcRiiAlLn8GlV7Ze/RjZBiFjDsW94HmGv6cHHmpwrv+zDn8CscdmNLcYzsYabIf67p0SZGY9Brs8Z/0Y3wm8zc/wHsMQaZth3GXdLuTROM0oZMm4f8qBBeBH9lPXObSinuUe84U3KoxoXQa6tUc6SQ/58p5R2497hj/7Z5RxyMx9FlhvKCeSs5BxD3lytvgjydu4oK5C5ogs7cpCdPsPHzvbqCxKJRCLxN3gF7F+F9X4fnikAAAAASUVORK5CYII=>

[image64]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAZCAYAAABKM8wfAAABYklEQVR4Xu2VPytHURjHn/zJYPMCJEx4A5TNOzCgGBSlGCWbm6wWs0XZWKzeACmDTcokkQFRBgnfp3t+Ofd7z7n36ec3SOdTn+F8z3Oe89zlXJFE4n9wAU/hMpyDM3AaTjmZDvjOoeMLHsIlOCv1vZpCL4n55NVd0l4IPu975tX9Cm02DofhIOx3xoZ6kfjeJxyFQ2Lr1RTnHIATOMKhIzZwG9zmEDzCbg5byRg84NAjNnCINbjCYQD92F14A7fganG7mrphrAN3iq2uXYp1vK5kz1mFdeBrOM9hAH2duN8RraPowT4OCevAlhqlR/JadQf2FrfjLIjtEsvA+1Jf46NvdmNo9bm4HeZKbJdYBm5cbKGL1puSn80oL2G9pNUDb8B1yvTve0xZCeslb1JfZ+2lZFKuvYcTlJXQQx8cerzCO8nfSvVW8p/CgF/k0F66byGDk/BBfj500S9IJBKJxN/gGyV9cgESwUd1AAAAAElFTkSuQmCC>

[image65]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAZCAYAAABnweOlAAAC6klEQVR4Xu2XS8hNURTH/155RB4ToQzIxJuxJBRJCBMDKRNMPMsXGUjMlCgKpS+vgZRH3pQSUjJUxsj7/czb+lt7u+uus+8515WB2r/69539X2vvs/c6d+9zPiCTyWT+DWNEz0U/RDdEferDv7kAzXkpWuxi5IRol2ioqAN03HOisTbJsUf0Djou9ak+XOAparkfRGdFo0SvjE+9ET0TvQ3tz6JhaJKloh2mfRA6yHjjEXpdwvWC0H5SC//iavCtDtVlNOaK6Bi0TxksPHO4YM9MaGylD0DnythoH0gRJ1/m8WkcN21yGpozy3iXRatEB0RrjN8MLEpn6JjbXSzSBn0wVUVZ5gPCQBTX1ZD7KCb6zl9De77xRgTvsfEuQbdNK7AoxN/bEv2qoizxAaE3yscuZT204wzjDRLtN20yCZp3y3gX8fdF4XbmuANMjHQXLQ/XrRTlOjQ2xQeqmAPt2OjnazkPzR3pvBXQQ609xLnIZohFIex3x7QJCx6pKsoG6GHPg5WHMF8i31B+4CfZKjoC7VxVzU7Qm990/knRRtPuBs2barxG2KLcRfFn/tBcVxVlm2gydB3TRZuCv7mW+mdwq3CAUz5g+Ij6bVMGx/ILTGGLwqfMPgtDe6JoeC1cWZTU9ukFjT3ygWYpW8ht6C8qRXxlW8rGstiiEPb5Hq75rWFppSgkzmWwD3i4XfY6L3ae4Pyjoi3Ouxf+8kbsw1e1pdWi7IP248F92MWqitLoHItzmesDlnlITzp6PDsia1F8//cX7QzXQ6B9/PlBj6/0KnxRCPvyq7Vjwi8rip9nJLXWJEzqatr84qN3xng8sOKAXtNMHtt2AeuCxwO3jB7QvHbn81cct1AkfuC9dj5ZBI2tdn5f6BuRsdkulqQf9MYUq8+O8elHfCGsbBF4zZt/MfGeJp6C/wbw8OM25IfkCxPjk7eL4P8zD1DL5YfjNdE4FOfFgnIu76Gv5N3QgmYymUwmk8n8l/wEyn38mh8mTOAAAAAASUVORK5CYII=>

[image66]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAAb0lEQVR4XmNgGAUDBfiA2B2IvdAwCpAH4v94cDpMIS9UoAAmAASvgPgfEh8OQAo3oImlQMVRAMg9GIJAsIIBi/gObIIMELEP6ILLoBLIQAQqBgoZFCAMlYABJig/CkkMBdgwIILoJhALokqPAkwAAM/RHmAMsOQ3AAAAAElFTkSuQmCC>

[image67]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAWCAYAAAAW5GZjAAAAiElEQVR4XmNgGJTAAl0AH0gF4v9AvB1dAhd4yQDRQBQwYYAoDkWXwAVAin+hC+ICRxhIcIoQA0RxHboELgBSTNB0OQaIIjco7YUqjQAaDBAF7FA+iP0VIY0AxgwQSWUksfNQMRRgDRUMRhMXh4qXwAQ4oQJTYAJoAMOjvsgcNMAGxFnogsMSAADulx1sUN1HDAAAAABJRU5ErkJggg==>

[image68]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAWCAYAAAArdgcFAAABI0lEQVR4Xu2UP0tCURjGTy2SbTU49B1CAglBQvATGNYS0tbs4CRNzk1+CXFzcmhtrM2lNYyG9iAI/7yP57mX97wc6eIiwv3Bg+f9vee+Xu496lzOLjmWPEuWklfJQdjOzEAy1eLM+aFHrE9ZH6Y7slF0/rpg+I9kpIXwJvk17j8WLjIc4lYLoUeflSdJxZnhVxS1RJB7+hPjYxQk71wHwzsUF4kgN/SXxsf4U+tgeJ/iPG17mvR3xlseJXVVB8MfKMpp29OibxivwWn6MC76zKtp29OmxzHdBE6ZJRiOlwGxzWl5iQTX4EuxXgOBX5ZmQq/BSy4ZZwnuHMTuEvW1qvF3AGf3WdD/tHIomfMTG3BELWNJ10qCR/ElmTk//Dts5+w9K6T/T6CXjpHfAAAAAElFTkSuQmCC>

[image69]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAXCAYAAAAyet74AAAAmUlEQVR4XmNgGDSAEYhV0QXRwVMg/g/FBMEVBiIVghRdQxfEBkAKI9AF0UEUA6a1TUDsjybGcJMBoZALiO8DMR8Qf4OrgAKQottALAjEG6FiP6HiKAAksBOIZ6JLIAOQB0AKr0LpPajSCHCdAdUKEHsKEh8O0MMPxF8JZX9EEgdLhKHxsxkgcX8MJigGlUAGflCxD2jiIwwAAH0kJ3M+/9rpAAAAAElFTkSuQmCC>

[image70]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAWCAYAAAASEbZeAAAAjUlEQVR4XmNgGNxAH4jfAvF/ID4BxAKo0gwMGUA8CYm/hAGi2AhJDCwAwnjFnqALQPnoYiigigGiwAtdAgYCGCAKJqJLwEAPEK8C4r9A7IwmhwGkGSCmbUGXQAcYDgcZPxtZgAGhyAbECUYSQAYwMWZkAXa4NAODHlRsG5IYgxAQ/4PiN1AFU5EVDEkAANzBJyD66jd7AAAAAElFTkSuQmCC>

[image71]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAWCAYAAAASEbZeAAAAbElEQVR4XmNgGDpAHoj/owuiA5ACvIp2AfFXBjyKZIB4AxC/YcCjCCaBU9FaIJaFsrEqEgbifUh8rIrQBTAUzQRiDWQBBiyKtgPxYTQMCycQew5CKSogGJgggFfROSB+AcSPoRjEPo2iYigBALmyJr5sPTF+AAAAAElFTkSuQmCC>