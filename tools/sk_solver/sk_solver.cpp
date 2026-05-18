// CGAL weighted straight skeleton CLI wrapper.
//
// Reads JSON from stdin:
//   {
//     "polygon": [[x0, z0], [x1, z1], ...],   // CCW outer contour
//     "weights": [w0, w1, ...]                  // one per edge (between vertex i and vertex i+1)
//   }
//
// Writes JSON to stdout:
//   {
//     "ok": true,
//     "input_count": N,
//     "faces": [
//       {
//         "edge_index": i,                      // which input edge this face starts on
//         "vertices": [
//           {"x": ..., "z": ..., "time": ...},  // time = skeleton distance (0 on boundary)
//           ...
//         ]
//       },
//       ...
//     ]
//   }
//
// Errors return {"ok": false, "error": "..."} to stdout (exit 1).
//
// Build: cmake-driven; see CMakeLists.txt in the same directory.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Straight_skeleton_2.h>
#include <CGAL/create_weighted_straight_skeleton_2.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point2 = K::Point_2;
using Polygon2 = CGAL::Polygon_2<K>;
using Skeleton = CGAL::Straight_skeleton_2<K>;
using SkeletonPtr = std::shared_ptr<Skeleton>;

namespace {

// Tiny JSON helpers — we write the output ourselves and parse a fixed input
// schema, so a full JSON library is overkill. Input parser tolerates whitespace
// and trailing commas.

std::string slurp_stdin() {
    std::ostringstream oss;
    oss << std::cin.rdbuf();
    return oss.str();
}

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
        ++i;
    }
}

bool consume(const std::string& s, size_t& i, char c) {
    skip_ws(s, i);
    if (i < s.size() && s[i] == c) {
        ++i;
        return true;
    }
    return false;
}

bool parse_number(const std::string& s, size_t& i, double& out) {
    skip_ws(s, i);
    size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
    while (i < s.size() && (isdigit(s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
        ++i;
    }
    if (i == start) return false;
    try {
        out = std::stod(s.substr(start, i - start));
    } catch (...) {
        return false;
    }
    return true;
}

bool parse_pair(const std::string& s, size_t& i, double& a, double& b) {
    if (!consume(s, i, '[')) return false;
    if (!parse_number(s, i, a)) return false;
    if (!consume(s, i, ',')) return false;
    if (!parse_number(s, i, b)) return false;
    return consume(s, i, ']');
}

bool parse_input(const std::string& s, std::vector<Point2>& polygon, std::vector<double>& weights) {
    size_t i = 0;
    if (!consume(s, i, '{')) return false;
    bool got_polygon = false, got_weights = false;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == '}') break;
        if (!consume(s, i, '"')) return false;
        size_t key_start = i;
        while (i < s.size() && s[i] != '"') ++i;
        std::string key = s.substr(key_start, i - key_start);
        if (!consume(s, i, '"')) return false;
        if (!consume(s, i, ':')) return false;
        if (key == "polygon") {
            if (!consume(s, i, '[')) return false;
            while (true) {
                skip_ws(s, i);
                if (i < s.size() && s[i] == ']') { ++i; break; }
                double x, z;
                if (!parse_pair(s, i, x, z)) return false;
                polygon.emplace_back(x, z);
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',') ++i;
            }
            got_polygon = true;
        } else if (key == "weights") {
            if (!consume(s, i, '[')) return false;
            while (true) {
                skip_ws(s, i);
                if (i < s.size() && s[i] == ']') { ++i; break; }
                double w;
                if (!parse_number(s, i, w)) return false;
                weights.push_back(w);
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',') ++i;
            }
            got_weights = true;
        } else {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') ++i;
    }
    return got_polygon && got_weights;
}

void emit_error(const std::string& msg) {
    std::cout << "{\"ok\":false,\"error\":\"" << msg << "\"}";
}

} // namespace

int main() {
    std::string raw = slurp_stdin();
    std::vector<Point2> polygon;
    std::vector<double> weights;
    if (!parse_input(raw, polygon, weights)) {
        emit_error("invalid input json");
        return 1;
    }
    if (polygon.size() < 3) {
        emit_error("polygon needs at least 3 vertices");
        return 1;
    }
    if (weights.size() != polygon.size()) {
        emit_error("weights length must equal polygon length (one weight per edge)");
        return 1;
    }

    // CGAL's create_interior_weighted_straight_skeleton_2 wants holes; we have
    // none, so pass empty hole iterators.
    std::vector<Polygon2> empty_holes;
    std::vector<std::vector<double>> empty_hole_weights;

    SkeletonPtr sk;
    try {
        sk = CGAL::create_interior_weighted_straight_skeleton_2(
            polygon.begin(), polygon.end(),
            empty_holes.begin(), empty_holes.end(),
            weights.begin(), weights.end(),
            empty_hole_weights.begin(), empty_hole_weights.end()
        );
    } catch (const std::exception& e) {
        emit_error(std::string("CGAL exception: ") + e.what());
        return 1;
    }

    if (!sk) {
        emit_error("CGAL returned null skeleton");
        return 1;
    }

    // Map each contour halfedge to its input-edge index. The skeleton's
    // contour halfedges correspond to the input polygon edges in order; the
    // ID() / id() of the *vertex* at the start of the contour halfedge tells
    // us the input vertex index.
    std::cout << "{\"ok\":true,";
    std::cout << "\"input_count\":" << polygon.size() << ",";
    std::cout << "\"faces\":[";

    bool first_face = true;
    for (auto fit = sk->faces_begin(); fit != sk->faces_end(); ++fit) {
        // Each face is bounded by halfedges. The "defining contour" halfedge
        // is the one whose opposite is on the polygon boundary (is_border).
        // Walk all halfedges of this face once to find it.
        auto h_start = fit->halfedge();
        if (h_start == nullptr) continue;

        // Find the contour halfedge of this face: a halfedge whose `is_bisector` is false
        // (it lies on the polygon contour).
        auto h_contour = h_start;
        {
            auto h = h_start;
            do {
                if (!h->is_bisector()) {
                    h_contour = h;
                    break;
                }
                h = h->next();
            } while (h != h_start);
        }

        // The contour halfedge's source vertex is the polygon vertex i
        // (where edge i goes from vertex i to vertex i+1). We use that vertex's
        // id() to identify the input edge.
        int edge_index = -1;
        if (h_contour->vertex() != nullptr) {
            // h_contour goes from prev vertex to this vertex; the EDGE i
            // corresponds to source vertex.id() (the source).
            auto src_v = h_contour->opposite()->vertex();
            if (src_v != nullptr) {
                edge_index = static_cast<int>(src_v->id());
            }
        }

        if (!first_face) std::cout << ",";
        first_face = false;
        std::cout << "{\"edge_index\":" << edge_index << ",\"vertices\":[";

        // Walk the face boundary, collecting vertex (x, z, time) tuples.
        bool first_v = true;
        auto h = h_start;
        do {
            auto v = h->vertex();
            if (v != nullptr) {
                if (!first_v) std::cout << ",";
                first_v = false;
                const auto& p = v->point();
                std::cout << "{\"x\":" << p.x() << ",\"z\":" << p.y()
                          << ",\"time\":" << v->time() << "}";
            }
            h = h->next();
        } while (h != h_start);

        std::cout << "]}";
    }

    std::cout << "]}";
    std::cout << std::endl;
    return 0;
}
