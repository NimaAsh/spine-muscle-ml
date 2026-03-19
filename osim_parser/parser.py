from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import xml.etree.ElementTree as ET


def _text_to_floats(text: Optional[str]) -> List[float]:
    if not text:
        return []
    # Split on whitespace, allow things like "-0" and scientific notation
    parts = re.split(r"\s+", text.strip())
    values: List[float] = []
    for token in parts:
        if token == "":
            continue
        try:
            values.append(float(token))
        except ValueError:
            # Keep as-is when not a float
            return []
    return values


def _get_text(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None:
        return None
    value = elem.text.strip() if elem.text else None
    return value


def _maybe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---- Math utilities (no numpy) ----

def _rx(angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _ry(angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rz(angle: float) -> List[List[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _mat3_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mat3_vec(a: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _mat3_T(a: List[List[float]]) -> List[List[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _axis_angle(axis: List[float], angle: float) -> List[List[float]]:
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0.0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x /= n
    y /= n
    z /= n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    return [
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ]


def _euler_xyz_body_fixed_to_R(angles: List[float]) -> List[List[float]]:
    if not angles:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rx = angles[0] if len(angles) > 0 else 0.0
    ry = angles[1] if len(angles) > 1 else 0.0
    rz = angles[2] if len(angles) > 2 else 0.0
    # Body-fixed XYZ: composite R = Rz(rz) * Ry(ry) * Rx(rx)
    return _mat3_mul(_mat3_mul(_rz(rz), _ry(ry)), _rx(rx))


def _R_to_euler_xyz_body_fixed(R: List[List[float]]) -> List[float]:
    # Inverse of R = Rz(rz) * Ry(ry) * Rx(rx)
    # ry = asin(-R[2][0])
    if abs(R[2][0]) < 1.0:
        ry = math.asin(-R[2][0])
        rx = math.atan2(R[2][1], R[2][2])
        rz = math.atan2(R[1][0], R[0][0])
    else:
        # Gimbal lock
        ry = math.copysign(math.pi / 2, -R[2][0])
        rx = 0.0
        rz = math.atan2(-R[0][1], R[1][1])
    return [rx, ry, rz]


def _homogeneous(R: List[List[float]], p: List[float]) -> List[List[float]]:
    return [
        [R[0][0], R[0][1], R[0][2], p[0]],
        [R[1][0], R[1][1], R[1][2], p[1]],
        [R[2][0], R[2][1], R[2][2], p[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _mat4_inv(X: List[List[float]]) -> List[List[float]]:
    R = [[X[i][j] for j in range(3)] for i in range(3)]
    p = [X[i][3] for i in range(3)]
    RT = _mat3_T(R)
    mRp = _mat3_vec(RT, [-p[0], -p[1], -p[2]])
    return _homogeneous(RT, mRp)


def _child(element: ET.Element, tag: str) -> Optional[ET.Element]:
    return element.find(tag)


def _children(element: ET.Element, tag: str) -> List[ET.Element]:
    return list(element.findall(tag))


def _attrs(element: ET.Element) -> Dict[str, str]:
    return dict(element.attrib) if element is not None else {}


def _parse_coordinate(coord_el: ET.Element) -> Dict[str, Any]:
    name = coord_el.attrib.get("name", "")
    d: Dict[str, Any] = {
        "name": name,
        "motion_type": _get_text(_child(coord_el, "motion_type")),
        "default_value": _maybe_float(_get_text(_child(coord_el, "default_value"))),
        "default_speed_value": _maybe_float(_get_text(_child(coord_el, "default_speed_value"))),
        "range": _text_to_floats(_get_text(_child(coord_el, "range"))),
        "clamped": (_get_text(_child(coord_el, "clamped")) == "true"),
        "locked": (_get_text(_child(coord_el, "locked")) == "true"),
        "prescribed": (_get_text(_child(coord_el, "prescribed")) == "true"),
    }
    return d


def _parse_transform_axis(axis_el: ET.Element) -> Dict[str, Any]:
    name = axis_el.attrib.get("name", "")
    coords = _get_text(_child(axis_el, "coordinates"))
    axis = _text_to_floats(_get_text(_child(axis_el, "axis")))
    # Extract function type and parameters crudely (LinearFunction / MultiplierFunction(Constant, scale), etc.)
    fn_el = axis_el.find("function")
    function: Dict[str, Any] = {}
    if fn_el is not None and len(fn_el):
        inner = list(fn_el)[0]
        fn_type = inner.tag
        function["type"] = fn_type
        if fn_type == "LinearFunction":
            function["coefficients"] = _text_to_floats(_get_text(inner.find("coefficients")))
        elif fn_type == "MultiplierFunction":
            function["scale"] = _maybe_float(_get_text(inner.find("scale")))
            inner_fn_el = inner.find("function")
            if inner_fn_el is not None and len(inner_fn_el):
                inner_inner = list(inner_fn_el)[0]
                function["inner_function"] = {
                    "type": inner_inner.tag,
                    "value": _maybe_float(_get_text(inner_inner.find("value"))),
                }
    return {
        "name": name,
        "coordinates": coords.split() if coords else [],
        "axis": axis,
        "function": function,
    }


def _parse_custom_joint(joint_el: ET.Element) -> Dict[str, Any]:
    name = joint_el.attrib.get("name", "")
    parent_body = _get_text(_child(joint_el, "parent_body"))
    location_in_parent = _text_to_floats(_get_text(_child(joint_el, "location_in_parent")))
    orientation_in_parent = _text_to_floats(_get_text(_child(joint_el, "orientation_in_parent")))
    location = _text_to_floats(_get_text(_child(joint_el, "location")))
    orientation = _text_to_floats(_get_text(_child(joint_el, "orientation")))
    reverse = _get_text(_child(joint_el, "reverse")) == "true"

    # Coordinates
    coordinates: Dict[str, Dict[str, Any]] = {}
    coordset = joint_el.find("CoordinateSet/objects")
    if coordset is not None:
        for coord in coordset.findall("Coordinate"):
            c = _parse_coordinate(coord)
            coordinates[c["name"]] = c

    # SpatialTransform
    st = joint_el.find("SpatialTransform")
    axes: List[Dict[str, Any]] = []
    if st is not None:
        for ax in st.findall("TransformAxis"):
            axes.append(_parse_transform_axis(ax))

    return {
        "type": "CustomJoint",
        "name": name,
        "parent_body": parent_body,
        "location_in_parent": location_in_parent,
        "orientation_in_parent": orientation_in_parent,
        "location": location,
        "orientation": orientation,
        "reverse": reverse,
        "coordinates": coordinates,
        "spatial_transform": axes,
    }


def _constant_linear_function() -> Dict[str, Any]:
    return {"type": "LinearFunction", "coefficients": [0.0, 0.0]}


def _parse_pin_joint(joint_el: ET.Element) -> Dict[str, Any]:
    name = joint_el.attrib.get("name", "")
    parent_body = _get_text(_child(joint_el, "parent_body"))
    location_in_parent = _text_to_floats(_get_text(_child(joint_el, "location_in_parent")))
    orientation_in_parent = _text_to_floats(_get_text(_child(joint_el, "orientation_in_parent")))
    location = _text_to_floats(_get_text(_child(joint_el, "location")))
    orientation = _text_to_floats(_get_text(_child(joint_el, "orientation")))
    reverse = _get_text(_child(joint_el, "reverse")) == "true"

    coordinates: Dict[str, Dict[str, Any]] = {}
    coordset = joint_el.find("CoordinateSet/objects")
    if coordset is not None:
        for coord in coordset.findall("Coordinate"):
            c = _parse_coordinate(coord)
            coordinates[c["name"]] = c

    # Normalize to 6 DoF spatial transform (OpenSim PinJoint rotates about Z)
    pin_coord_name = next(iter(coordinates.keys()), None)
    spatial_axes: List[Dict[str, Any]] = []
    # rotation1 about Z
    spatial_axes.append({
        "name": "rotation1",
        "coordinates": [pin_coord_name] if pin_coord_name else [],
        "axis": [0.0, 0.0, 1.0],
        "function": {"type": "LinearFunction", "coefficients": [1.0, 0.0]} if pin_coord_name else _constant_linear_function(),
    })
    # rotation2 about X (0)
    spatial_axes.append({
        "name": "rotation2",
        "coordinates": [],
        "axis": [1.0, 0.0, 0.0],
        "function": _constant_linear_function(),
    })
    # rotation3 about Y (0)
    spatial_axes.append({
        "name": "rotation3",
        "coordinates": [],
        "axis": [0.0, 1.0, 0.0],
        "function": _constant_linear_function(),
    })
    # translations all zero
    spatial_axes.append({
        "name": "translation1",
        "coordinates": [],
        "axis": [1.0, 0.0, 0.0],
        "function": _constant_linear_function(),
    })
    spatial_axes.append({
        "name": "translation2",
        "coordinates": [],
        "axis": [0.0, 1.0, 0.0],
        "function": _constant_linear_function(),
    })
    spatial_axes.append({
        "name": "translation3",
        "coordinates": [],
        "axis": [0.0, 0.0, 1.0],
        "function": _constant_linear_function(),
    })

    return {
        "type": "PinJoint",
        "name": name,
        "parent_body": parent_body,
        "location_in_parent": location_in_parent,
        "orientation_in_parent": orientation_in_parent,
        "location": location,
        "orientation": orientation,
        "reverse": reverse,
        "coordinates": coordinates,
        "spatial_transform": spatial_axes,
    }


def _parse_weld_joint(joint_el: ET.Element) -> Dict[str, Any]:
    name = joint_el.attrib.get("name", "")
    parent_body = _get_text(_child(joint_el, "parent_body"))
    location_in_parent = _text_to_floats(_get_text(_child(joint_el, "location_in_parent")))
    orientation_in_parent = _text_to_floats(_get_text(_child(joint_el, "orientation_in_parent")))
    location = _text_to_floats(_get_text(_child(joint_el, "location")))
    orientation = _text_to_floats(_get_text(_child(joint_el, "orientation")))
    reverse = _get_text(_child(joint_el, "reverse")) == "true"

    spatial_axes: List[Dict[str, Any]] = [
        {"name": "rotation1", "coordinates": [], "axis": [0.0, 0.0, 1.0], "function": _constant_linear_function()},
        {"name": "rotation2", "coordinates": [], "axis": [1.0, 0.0, 0.0], "function": _constant_linear_function()},
        {"name": "rotation3", "coordinates": [], "axis": [0.0, 1.0, 0.0], "function": _constant_linear_function()},
        {"name": "translation1", "coordinates": [], "axis": [1.0, 0.0, 0.0], "function": _constant_linear_function()},
        {"name": "translation2", "coordinates": [], "axis": [0.0, 1.0, 0.0], "function": _constant_linear_function()},
        {"name": "translation3", "coordinates": [], "axis": [0.0, 0.0, 1.0], "function": _constant_linear_function()},
    ]

    return {
        "type": "WeldJoint",
        "name": name,
        "parent_body": parent_body,
        "location_in_parent": location_in_parent,
        "orientation_in_parent": orientation_in_parent,
        "location": location,
        "orientation": orientation,
        "reverse": reverse,
        "coordinates": {},
        "spatial_transform": spatial_axes,
    }


def _parse_body(body_el: ET.Element) -> Dict[str, Any]:
    name = body_el.attrib.get("name", "")
    joint: Optional[Dict[str, Any]] = None
    joint_container = body_el.find("Joint")
    if joint_container is not None:
        cj = joint_container.find("CustomJoint")
        pj = joint_container.find("PinJoint")
        wj = joint_container.find("WeldJoint")
        if cj is not None:
            joint = _parse_custom_joint(cj)
        elif pj is not None:
            joint = _parse_pin_joint(pj)
        elif wj is not None:
            joint = _parse_weld_joint(wj)
        # Could add other joint types here if present

    visible_obj = body_el.find("VisibleObject")
    display: Dict[str, Any] = {}
    if visible_obj is not None:
        geomset = visible_obj.find("GeometrySet/objects")
        geometries: List[Dict[str, Any]] = []
        if geomset is not None:
            for dg in geomset.findall("DisplayGeometry"):
                geometries.append({
                    "geometry_file": _get_text(dg.find("geometry_file")),
                    "color": _text_to_floats(_get_text(dg.find("color"))),
                    "texture_file": _get_text(dg.find("texture_file")),
                    "transform": _text_to_floats(_get_text(dg.find("transform"))),
                    "scale_factors": _text_to_floats(_get_text(dg.find("scale_factors"))),
                    "display_preference": _maybe_float(_get_text(dg.find("display_preference"))),
                    "opacity": _maybe_float(_get_text(dg.find("opacity"))),
                })
        display = {
            "geometries": geometries,
            "scale_factors": _text_to_floats(_get_text(visible_obj.find("scale_factors"))),
            "transform": _text_to_floats(_get_text(visible_obj.find("transform"))),
            "show_axes": _get_text(visible_obj.find("show_axes")) == "true",
            "display_preference": _maybe_float(_get_text(visible_obj.find("display_preference"))),
        }

    # Wrap objects
    wrap_objects: List[Dict[str, Any]] = []
    wrap_set_objects = body_el.find("WrapObjectSet/objects")
    if wrap_set_objects is not None:
        for wrap_el in list(wrap_set_objects):
            w = {"type": wrap_el.tag, "name": wrap_el.attrib.get("name", "")}
            # Common properties
            w["xyz_body_rotation"] = _text_to_floats(_get_text(wrap_el.find("xyz_body_rotation")))
            w["translation"] = _text_to_floats(_get_text(wrap_el.find("translation")))
            w["active"] = (_get_text(wrap_el.find("active")) == "true")
            w["quadrant"] = _get_text(wrap_el.find("quadrant"))
            dims = _text_to_floats(_get_text(wrap_el.find("dimensions")))
            if dims:
                w["dimensions"] = dims
            wrap_objects.append(w)

    body_dict: Dict[str, Any] = {
        "name": name,
        "mass": _maybe_float(_get_text(body_el.find("mass"))),
        "mass_center": _text_to_floats(_get_text(body_el.find("mass_center"))),
        "inertias": {
            "xx": _maybe_float(_get_text(body_el.find("inertia_xx"))),
            "yy": _maybe_float(_get_text(body_el.find("inertia_yy"))),
            "zz": _maybe_float(_get_text(body_el.find("inertia_zz"))),
            "xy": _maybe_float(_get_text(body_el.find("inertia_xy"))),
            "xz": _maybe_float(_get_text(body_el.find("inertia_xz"))),
            "yz": _maybe_float(_get_text(body_el.find("inertia_yz"))),
        },
        "joint": joint,
        "visible": display,
        "wrap_objects": wrap_objects,
    }
    return body_dict


def _parse_pathpoint(pp_el: ET.Element) -> Dict[str, Any]:
    return {
        "name": pp_el.attrib.get("name", ""),
        "location": _text_to_floats(_get_text(pp_el.find("location"))),
        "body": _get_text(pp_el.find("body")),
    }


def _parse_muscle(muscle_el: ET.Element) -> Dict[str, Any]:
    mtype = muscle_el.tag
    name = muscle_el.attrib.get("name", "")

    path_pts: List[Dict[str, Any]] = []
    path_objects = muscle_el.find("GeometryPath/PathPointSet/objects")
    if path_objects is not None:
        for pp in path_objects.findall("PathPoint"):
            path_pts.append(_parse_pathpoint(pp))

    muscle_dict: Dict[str, Any] = {
        "type": mtype,
        "name": name,
        "path_points": path_pts,
        "max_isometric_force": _maybe_float(_get_text(muscle_el.find("max_isometric_force"))),
        "optimal_fiber_length": _maybe_float(_get_text(muscle_el.find("optimal_fiber_length"))),
        "tendon_slack_length": _maybe_float(_get_text(muscle_el.find("tendon_slack_length"))),
        "pennation_angle_at_optimal": _maybe_float(_get_text(muscle_el.find("pennation_angle_at_optimal"))),
    }
    return muscle_dict


def _parse_force_set(root: ET.Element) -> Dict[str, Dict[str, Any]]:
    forces: Dict[str, Dict[str, Any]] = {}
    fset = root.find(".//ForceSet/objects")
    if fset is None:
        return forces
    for el in list(fset):
        # Many muscle/actuator types; handle generically and add specifics above as needed
        m = _parse_muscle(el)
        forces[m["name"]] = m
    return forces


def _parse_groups(root: ET.Element) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    groups = root.find(".//groups")
    if groups is None:
        return mapping
    for g in groups.findall("ObjectGroup"):
        name = g.attrib.get("name", "")
        members_txt = _get_text(g.find("members")) or ""
        members = members_txt.split()
        mapping[name] = members
    return mapping


def _parse_bodies(root: ET.Element) -> Dict[str, Dict[str, Any]]:
    bodies: Dict[str, Dict[str, Any]] = {}
    for body in root.findall(".//Body"):
        bd = _parse_body(body)
        bodies[bd["name"]] = bd
    return bodies


def _parse_model_metadata(model_el: ET.Element) -> Dict[str, Any]:
    return {
        "name": model_el.attrib.get("name", ""),
        "credits": _get_text(model_el.find("credits")),
        "publications": _get_text(model_el.find("publications")),
        "length_units": _get_text(model_el.find("length_units")),
        "force_units": _get_text(model_el.find("force_units")),
        "gravity": _text_to_floats(_get_text(model_el.find("gravity"))),
    }


def parse_osim(path: str | Path) -> Dict[str, Any]:
    """Parse an OpenSim .osim file into a dictionary structure.

    Returns a dict with keys:
    - meta: model metadata
    - bodies: { body_name: {mass, mass_center, inertias, joint, visible} }
    - forces: { force_name: {type, path_points, scalar params...} }
    - groups: { group_name: [members...] }
    """
    xml_path = Path(path)
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    model_el = root.find(".//Model")
    if model_el is None:
        raise ValueError("No <Model> element found in OSIM file")

    meta = _parse_model_metadata(model_el)
    bodies = _parse_bodies(model_el)
    forces = _parse_force_set(model_el)
    groups = _parse_groups(model_el)

    return {
        "meta": meta,
        "bodies": bodies,
        "forces": forces,
        "groups": groups,
    }


def get_body_6dof(model: Dict[str, Any], body_name: str) -> Dict[str, Any]:
    """Return a normalized 6-DoF description for `body_name` relative to its parent.

    Output:
    - parent_body
    - location_in_parent: [x,y,z]
    - orientation_in_parent: [rx,ry,rz]
    - location_child: [x,y,z]
    - orientation_child: [rx,ry,rz]
    - axes: list of 6 transform axes (rotation1..3, translation1..3) with axis vectors and scalar function description
    - coordinates: mapping of coordinate name -> coord dict (range, defaults, etc.)
    """
    bodies = model.get("bodies", {})
    if body_name not in bodies:
        raise KeyError(f"Body not found: {body_name}")
    body = bodies[body_name]
    joint = body.get("joint") or {}

    result = {
        "parent_body": joint.get("parent_body"),
        "location_in_parent": joint.get("location_in_parent", []),
        "orientation_in_parent": joint.get("orientation_in_parent", []),
        "location_child": joint.get("location", []),
        "orientation_child": joint.get("orientation", []),
        "axes": joint.get("spatial_transform", []),
        "coordinates": joint.get("coordinates", {}),
        "joint_type": joint.get("type"),
        "joint_name": joint.get("name"),
        "reverse": joint.get("reverse", False),
    }
    return result


def get_all_bodies_6dof(model: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {name: get_body_6dof(model, name) for name in model.get("bodies", {}).keys()}


class OSIMModel:
    """Parsed OSIM + original XML tree for round-trip editing and save."""

    def __init__(self, path: Path, tree: ET.ElementTree, root: ET.Element, data: Dict[str, Any]):
        self.path = Path(path)
        self.tree = tree
        self.root = root
        self.data = data

    @classmethod
    def from_file(cls, path: str | Path) -> "OSIMModel":
        p = Path(path)
        tree = ET.parse(str(p))
        root = tree.getroot()
        data = parse_osim(p)
        return cls(p, tree, root, data)

    def set_muscle_max_isometric_force(self, muscle_name: str, value: float) -> None:
        # Update XML
        obj = self.root.find(f".//ForceSet/objects/*[@name='{muscle_name}']")
        if obj is None:
            raise KeyError(f"Muscle not found: {muscle_name}")
        maxf = obj.find("max_isometric_force")
        if maxf is None:
            # Some actuators may differ; create if missing
            maxf = ET.SubElement(obj, "max_isometric_force")
        maxf.text = str(value)
        # Update dict
        if muscle_name in self.data.get("forces", {}):
            self.data["forces"][muscle_name]["max_isometric_force"] = float(value)

    def save(self, out_path: str | Path) -> None:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        self.tree.write(str(outp), encoding="UTF-8", xml_declaration=True)

    # ---- Muscle geometry editing ----
    def set_muscle_path_point_location(self, muscle_name: str, index: int, location: List[float]) -> None:
        obj = self.root.find(f".//ForceSet/objects/*[@name='{muscle_name}']")
        if obj is None:
            raise KeyError(f"Muscle not found: {muscle_name}")
        pps = obj.find("GeometryPath/PathPointSet/objects")
        if pps is None:
            raise KeyError(f"PathPointSet not found for muscle: {muscle_name}")
        pplist = pps.findall("PathPoint")
        if index < 0 or index >= len(pplist):
            raise IndexError(f"PathPoint index out of range for {muscle_name}: {index}")
        loc_el = pplist[index].find("location")
        if loc_el is None:
            loc_el = ET.SubElement(pplist[index], "location")
        loc_el.text = f" {location[0]} {location[1]} {location[2]}"
        # Update dict mirror
        mus = self.data.get("forces", {}).get(muscle_name)
        if mus and 0 <= index < len(mus.get("path_points", [])):
            mus["path_points"][index]["location"] = [float(location[0]), float(location[1]), float(location[2])]

    def set_muscle_optimal_fiber_length(self, muscle_name: str, value: float) -> None:
        """Set optimal_fiber_length for a muscle."""
        obj = self.root.find(f".//ForceSet/objects/*[@name='{muscle_name}']")
        if obj is None:
            raise KeyError(f"Muscle not found: {muscle_name}")
        ofl = obj.find("optimal_fiber_length")
        if ofl is None:
            ofl = ET.SubElement(obj, "optimal_fiber_length")
        ofl.text = str(value)
        # Update dict
        if muscle_name in self.data.get("forces", {}):
            self.data["forces"][muscle_name]["optimal_fiber_length"] = float(value)

    def set_muscle_tendon_slack_length(self, muscle_name: str, value: float) -> None:
        """Set tendon_slack_length for a muscle."""
        obj = self.root.find(f".//ForceSet/objects/*[@name='{muscle_name}']")
        if obj is None:
            raise KeyError(f"Muscle not found: {muscle_name}")
        tsl = obj.find("tendon_slack_length")
        if tsl is None:
            tsl = ET.SubElement(obj, "tendon_slack_length")
        tsl.text = str(value)
        # Update dict
        if muscle_name in self.data.get("forces", {}):
            self.data["forces"][muscle_name]["tendon_slack_length"] = float(value)

    def set_body_mass(self, body_name: str, value: float) -> None:
        """Set mass for a body."""
        obj = self.root.find(f".//BodySet/objects/Body[@name='{body_name}']")
        if obj is None:
            raise KeyError(f"Body not found: {body_name}")
        mass_el = obj.find("mass")
        if mass_el is None:
            mass_el = ET.SubElement(obj, "mass")
        mass_el.text = str(value)
        # Update dict
        if body_name in self.data.get("bodies", {}):
            self.data["bodies"][body_name]["mass"] = float(value)

    def calculate_muscle_length(self, muscle_name: str) -> float:
        """Calculate total muscle length as sum of distances between consecutive via points.

        Returns length in meters.
        """
        mus = self.data.get("forces", {}).get(muscle_name)
        if not mus:
            raise KeyError(f"Muscle not found: {muscle_name}")
        path_points = mus.get("path_points", [])
        if len(path_points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(path_points) - 1):
            p1 = path_points[i].get("location", [0.0, 0.0, 0.0])
            p2 = path_points[i + 1].get("location", [0.0, 0.0, 0.0])
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dz = p2[2] - p1[2]
            total_length += math.sqrt(dx*dx + dy*dy + dz*dz)

        return float(total_length)

    # -------- Kinematics helpers --------
    def evaluate_axis_value(self, axis: Dict[str, Any], qmap: Dict[str, float]) -> float:
        fn = axis.get("function", {})
        coords = axis.get("coordinates", [])
        t = fn.get("type") if isinstance(fn, dict) else None
        if t == "LinearFunction":
            coeffs = fn.get("coefficients") or []
            a = coeffs[0] if len(coeffs) > 0 else 0.0
            b = coeffs[1] if len(coeffs) > 1 else 0.0
            if len(coords) == 1:
                return a * qmap.get(coords[0], 0.0) + b
            # If multiple, sum the coordinates then apply (rare in supplied OSIM)
            return a * sum(qmap.get(c, 0.0) for c in coords) + b
        if t == "MultiplierFunction":
            scale = fn.get("scale") or 0.0
            inner = fn.get("inner_function") or {}
            if inner.get("type") == "Constant":
                return scale * (inner.get("value") or 0.0)
            return 0.0
        return 0.0

    def relative_transform_parent_child(self, body_name: str, qmap: Optional[Dict[str, float]] = None) -> List[List[float]]:
        if qmap is None:
            qmap = {}
        body = self.data["bodies"][body_name]
        joint = body.get("joint") or {}

        # Parent fixed placement
        Rp = _euler_xyz_body_fixed_to_R(joint.get("orientation_in_parent", []))
        pp = joint.get("location_in_parent", [0.0, 0.0, 0.0])
        X_P_Jp = _homogeneous(Rp, pp)

        # Axes (body fixed order rotation3*rotation2*rotation1 then translations)
        axes = joint.get("spatial_transform", [])
        # Compose rotational axes (3x3)
        R_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for ax in axes:
            if ax.get("name", "").startswith("rotation"):
                angle = self.evaluate_axis_value(ax, qmap)
                Ra = _axis_angle(ax.get("axis", [0.0, 0.0, 1.0]), angle)
                R_axes = _mat3_mul(Ra, R_axes)
        # Translations in child axes order
        t_vals = []
        for ax in axes:
            if ax.get("name", "").startswith("translation"):
                t_vals.append(self.evaluate_axis_value(ax, qmap))
        tx = t_vals[0] if len(t_vals) > 0 else 0.0
        ty = t_vals[1] if len(t_vals) > 1 else 0.0
        tz = t_vals[2] if len(t_vals) > 2 else 0.0
        # Express translation in rotated (child) frame for standard spatial transform ordering
        p_axes = [tx, ty, tz]
        X_axes = _homogeneous(R_axes, p_axes)

        # Child fixed offset (invert joint-in-child to go to child frame)
        Rc = _euler_xyz_body_fixed_to_R(joint.get("orientation", []))
        pc = joint.get("location", [0.0, 0.0, 0.0])
        X_Jc_C = _homogeneous(_mat3_T(Rc), _mat3_vec(_mat3_T(Rc), [-pc[0], -pc[1], -pc[2]]))

        return _mat4_mul(_mat4_mul(X_P_Jp, X_axes), X_Jc_C)

    def transform_relative_to(self, target_body: str, root_body: str, qmap: Optional[Dict[str, float]] = None) -> List[List[float]]:
        # Walk up from target to root collecting transforms
        if qmap is None:
            qmap = {}
        if target_body == root_body:
            return _homogeneous([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[0.0,0.0,0.0])
        bodies = self.data["bodies"]
        chain: List[str] = []
        cur = target_body
        visited = set()
        while cur and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            j = bodies[cur].get("joint") or {}
            parent = j.get("parent_body")
            if parent == root_body:
                chain.append(parent)
                break
            cur = parent
        # Compose from child->parent along chain
        X_root_target = _homogeneous([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[0.0,0.0,0.0])
        for i in range(len(chain)-1):
            child = chain[i]
            parent = bodies[child].get("joint", {}).get("parent_body")
            if parent is None:
                break
            X_parent_child = self.relative_transform_parent_child(child, qmap)
            X_root_target = _mat4_mul(X_root_target, X_parent_child)
            if parent == root_body:
                break
        return X_root_target


def _cli(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Parse and optionally modify an OpenSim .osim")
    parser.add_argument("osim", type=str, help="Path to .osim file")
    parser.add_argument("--section", choices=["all", "meta", "bodies", "forces", "groups"], default="all")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--body", type=str, default=None, help="If set, dump only this body's 6DoF info")
    parser.add_argument("--set-muscle-max-force", nargs=2, metavar=("MUSCLE_NAME", "VALUE"), help="Set max_isometric_force and write to --out")
    parser.add_argument("--out", type=str, default=None, help="Output .osim path when modifying")
    args = parser.parse_args(argv)

    model = OSIMModel.from_file(args.osim)

    if args.set_muscle_max_force:
        mus, val_str = args.set_muscle_max_force
        val = float(val_str)
        model.set_muscle_max_isometric_force(mus, val)
        if not args.out:
            raise SystemExit("--out is required when modifying")
        model.save(args.out)
        print(f"Wrote modified OSIM to: {args.out}")
        return 0

    # read-only dump
    data = model.data
    out = get_body_6dof(data, args.body) if args.body else (data if args.section == "all" else data[args.section])
    json.dump(out, sys.stdout, indent=args.indent)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))


