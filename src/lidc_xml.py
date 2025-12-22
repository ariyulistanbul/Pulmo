import os
import numpy as np
from lxml import etree

def parse_lidc_xml_rois(xml_path: str):
    """
    Parse LIDC XML and return list of nodules with ROI z-positions.
    Output list elements:
      {
        "nodule_id": str,
        "roi_z_positions": [float, ...],  # imageZposition values where ROI exists
        "all_points": [ (z, x, y), ... ]  # optional, not used for slice selection
      }
    This is robust-ish across schema variants.
    """
    if not os.path.isfile(xml_path):
        return []

    try:
        parser = etree.XMLParser(recover=True, huge_tree=True)
        tree = etree.parse(xml_path, parser)
        root = tree.getroot()
    except Exception:
        return []

    # Find all unblindedReadNodule nodes (can be nested)
    nodule_nodes = root.findall(".//unblindedReadNodule")
    out = []

    for i, nn in enumerate(nodule_nodes):
        nodule_id = nn.findtext(".//noduleID")
        if not nodule_id:
            nodule_id = f"nodule_{i}"

        roi_nodes = nn.findall(".//roi")
        z_list = []
        pts = []
        for roi in roi_nodes:
            z_txt = roi.findtext(".//imageZposition")
            if z_txt is None:
                # some variants might use imageZPosition or similar
                z_txt = roi.findtext(".//imageZPosition")
            if z_txt is None:
                continue
            try:
                z = float(z_txt)
            except Exception:
                continue

            z_list.append(z)

            # collect points (edgeMap)
            for em in roi.findall(".//edgeMap"):
                x_txt = em.findtext(".//xCoord")
                y_txt = em.findtext(".//yCoord")
                try:
                    x = int(float(x_txt))
                    y = int(float(y_txt))
                    pts.append((z, x, y))
                except Exception:
                    continue

        z_list = sorted(list(set(z_list)))
        out.append({
            "nodule_id": str(nodule_id),
            "roi_z_positions": z_list,
            "all_points": pts,
        })

    return out

def map_roi_z_to_slice_indices(roi_z_positions, dicom_z_positions: np.ndarray, max_dist=3.0):
    """
    roi_z_positions: list[float] from XML
    dicom_z_positions: (Z,) float
    returns: sorted unique slice indices
    max_dist: tolerance in same units as z positions (usually mm). LIDC typically aligns well.
    """
    if dicom_z_positions is None or len(dicom_z_positions) == 0:
        return []

    Z = len(dicom_z_positions)
    out = set()
    for z in roi_z_positions:
        # nearest index
        idx = int(np.argmin(np.abs(dicom_z_positions - z)))
        dist = float(np.abs(dicom_z_positions[idx] - z))
        if dist <= max_dist:
            out.add(idx)
        else:
            # even if far, still add nearest (some series have odd z metadata)
            out.add(idx)

    out = sorted([int(np.clip(i, 0, Z-1)) for i in out])
    return out
