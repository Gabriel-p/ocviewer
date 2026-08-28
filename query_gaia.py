import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

# Gaia EDR3 zero points. Sigmas are already squared here.
Zp_G, sigma_ZG_2 = 25.6873668671, 0.00000759
Zp_BP, sigma_ZBP_2 = 25.3385422158, 0.000007785
Zp_RP, sigma_ZRP_2 = 24.7478955012, 0.00001428


def query_run(
    c_ra: float,
    c_dec: float,
    box_s: float = 0.5,
    gaia_data_path: str = "/media/gabriel/backup/gabriel/GaiaDR3",
    max_mag: float = 19.0,
    plx_min: float = 0.0,
) -> pd.DataFrame:
    """
    Queries Gaia data frames based on specified parameters and returns a combined
    DataFrame.

    Parameters
    ----------
    c_ra : float
        Central right ascension for the query.
    c_dec : float
        Central declination for the query.
    box_s : float
        Size of the box to query (in degrees).
    frames_path : str
        Path to the directory containing Gaia data frames.
    max_mag : float
        Maximum magnitude for data retrieval.
    plx_min : float
        Minimum parallax value for data retrieval.

    Returns
    -------
    gaia_frame: pd.DataFrame
        DataFrame containing the combined Gaia data.
    """
    frames_path = gaia_data_path + "/datafiles_G20/"
    # Path to the file that informs the sky area covered by each raw data file
    path_gaia_frames_ranges = gaia_data_path + "/files_G20/frame_ranges.txt"

    # DataFrame containing information about Gaia data frames.
    fdata = pd.read_csv(path_gaia_frames_ranges)

    c_ra_l = [c_ra]
    if c_ra - box_s < 0:
        # Split frame, c_ra + 360
        c_ra_l.append(c_ra + 360)
    if c_ra > box_s > 360:
        # Split frame, c_ra - 360
        c_ra_l.append(c_ra - 360)

    dicts = []
    for c_ra in c_ra_l:
        data_in_files, xmin_cl, xmax_cl, ymin_cl, ymax_cl = findFrames(
            c_ra, c_dec, box_s, fdata
        )

        if len(data_in_files) == 0:
            continue

        all_frames = query(
            Zp_G,
            c_ra,
            c_dec,
            box_s,
            frames_path,
            max_mag,
            data_in_files,
            xmin_cl,
            xmax_cl,
            ymin_cl,
            ymax_cl,
            plx_min,
        )

        dicts.append(all_frames)

    if len(dicts) > 1:
        # Combine
        all_frames = (
            pd.concat([dicts[0], dicts[1]]).drop_duplicates().reset_index(drop=True)
        )
    else:
        all_frames = dicts[0]

    all_frames = uncertMags(
        Zp_G, Zp_BP, Zp_RP, sigma_ZG_2, sigma_ZBP_2, sigma_ZRP_2, all_frames
    )
    gaia_frame = all_frames.drop(columns=["FG", "e_FG", "FBP", "e_FBP", "FRP", "e_FRP"])

    # Round all values ('Source' can be included here since it is all ints)
    for col in gaia_frame.columns:
        gaia_frame[col] = gaia_frame[col].round(5)

    return pd.DataFrame(gaia_frame)


def findFrames(
    c_ra: float, c_dec: float, box_s_eq: float, fdata: pd.DataFrame
) -> tuple[list[str], float, float, float, float]:
    """
    Identifies Gaia data frames that overlap with the specified region.

    Parameters
    ----------
    c_ra : float
        Central right ascension of the region.
    c_dec : float
        Central declination of the region.
    box_s_eq : float
        Size of the box to query (in degrees).
    fdata : pd.DataFrame
        DataFrame containing information about Gaia data frames.

    Returns
    -------
    tuple
        A tuple containing:
        - data_in_files: List of filenames of overlapping frames.
        - xmin_cl: Minimum RA of the cluster region.
        - xmax_cl: Maximum RA of the cluster region.
        - ymin_cl: Minimum Dec of the cluster region.
        - ymax_cl: Maximum Dec of the cluster region.
    """
    # These are the points that determine the range of *all* the frames
    ra_min, ra_max = fdata["ra_min"].values, fdata["ra_max"].values
    dec_min, dec_max = fdata["dec_min"].values, fdata["dec_max"].values

    # frame == 'galactic':
    box_s_eq = np.sqrt(2) * box_s_eq
    # Correct size in RA
    box_s_x = box_s_eq / np.cos(np.deg2rad(c_dec))

    xl, yl = box_s_x * 0.5, box_s_eq * 0.5

    # Limits of the cluster's region in Equatorial
    xmin_cl, xmax_cl = c_ra - xl, c_ra + xl
    ymin_cl, ymax_cl = c_dec - yl, c_dec + yl

    frame_intersec = np.full(len(fdata), False)
    # Identify which frames contain the cluster region
    l2 = (xmin_cl, ymax_cl)  # Top left
    r2 = (xmax_cl, ymin_cl)  # Bottom right
    for i, xmin_fr_i in enumerate(ra_min):
        l1 = (xmin_fr_i, dec_max[i])  # Top left
        r1 = (ra_max[i], dec_min[i])  # Bottom right
        frame_intersec[i] = doOverlap(l1, r1, l2, r2)

    data_in_files = list(fdata[frame_intersec]["filename"])

    return data_in_files, xmin_cl, xmax_cl, ymin_cl, ymax_cl


def doOverlap(
    l1: tuple[float, float],
    r1: tuple[float, float],
    l2: tuple[float, float],
    r2: tuple[float, float],
) -> bool:
    """
    Checks if two rectangles defined by their top-left and bottom-right coordinates
    overlap.

    Source: https://www.geeksforgeeks.org/find-two-rectangles-overlap/

    Parameters
    ----------
    l1 : tuple
        Top-left coordinate of the first rectangle (x, y).
    r1 : tuple
        Bottom-right coordinate of the first rectangle (x, y).
    l2 : tuple
        Top-left coordinate of the second rectangle (x, y).
    r2 : tuple
        Bottom-right coordinate of the second rectangle (x, y).

    Returns
    -------
    bool
        True if the rectangles overlap, False otherwise.
    """
    min_x1, max_y1 = l1
    max_x1, min_y1 = r1
    min_x2, max_y2 = l2
    max_x2, min_y2 = r2
    # If one rectangle is on left side of other
    if min_x1 > max_x2 or min_x2 > max_x1:
        return False
    # If one rectangle is above other
    if min_y1 > max_y2 or min_y2 > max_y1:
        return False
    return True


def query(
    Zp_G: float,
    c_ra: float,
    c_dec: float,
    box_s_eq: float,
    frames_path: str,
    max_mag: float,
    data_in_files: list[str],
    xmin_cl: float,
    xmax_cl: float,
    ymin_cl: float,
    ymax_cl: float,
    plx_min: float,
) -> pd.DataFrame:
    """
    Queries individual Gaia data frames and combines the results.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    Zp_G : float
        Zero point for the G band.
    c_ra : float
        Central right ascension of the region.
    c_dec : float
        Central declination of the region.
    box_s_eq : float
        Size of the box to query (in degrees).
    frames_path : str
        Path to the directory containing Gaia data frames.
    max_mag : float
        Maximum magnitude for data retrieval.
    data_in_files : list
        List of filenames of overlapping frames.
    xmin_cl : float
        Minimum RA of the cluster region.
    xmax_cl : float
        Maximum RA of the cluster region.
    ymin_cl : float
        Minimum Dec of the cluster region.
    ymax_cl : float
        Maximum Dec of the cluster region.
    plx_min : float
        Minimum parallax value for data retrieval.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the combined Gaia data from the queried frames.
    """
    # Mag (flux) filter
    min_G_flux = 10 ** ((max_mag - Zp_G) / (-2.5))

    all_frames = []
    for file in data_in_files:
        data = pd.read_parquet(frames_path + file)

        mx = (data["ra"] >= xmin_cl) & (data["ra"] <= xmax_cl)
        my = (data["dec"] >= ymin_cl) & (data["dec"] <= ymax_cl)
        m_plx = data["parallax"] > plx_min
        m_gmag = data["phot_g_mean_flux"] > min_G_flux
        msk = mx & my & m_plx & m_gmag

        if msk.sum() == 0:
            continue
        # logging.info(f"N={msk.sum()} stars in {file}")

        all_frames.append(data[msk])
    all_frames = pd.concat(all_frames)

    c_ra, c_dec = c_ra, c_dec
    box_s_h = box_s_eq * 0.5

    gc = SkyCoord(ra=c_ra * u.degree, dec=c_dec * u.degree)  # pyright: ignore
    lb = gc.transform_to("galactic")
    gal_cent = (lb.l.value, lb.b.value)  # pyright: ignore

    if all_frames["l"].max() - all_frames["l"].min() > 180:
        # Fix frame that wraps around 360 in longitude
        lon = np.array(all_frames["l"])
        if gal_cent[0] > 180:
            lon[lon < 180] += 360
        else:
            lon[lon > 180] -= 360
        all_frames["l"] = lon

    xmin_cl, xmax_cl = gal_cent[0] - box_s_h, gal_cent[0] + box_s_h
    ymin_cl, ymax_cl = gal_cent[1] - box_s_h, gal_cent[1] + box_s_h
    mx = (all_frames["l"] >= xmin_cl) & (all_frames["l"] <= xmax_cl)
    my = (all_frames["b"] >= ymin_cl) & (all_frames["b"] <= ymax_cl)
    msk = mx & my
    all_frames = pd.DataFrame(all_frames[msk])

    all_frames = all_frames.rename(
        columns={
            "source_id": "Source",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "parallax": "Plx",
            "parallax_error": "e_Plx",
            "pmra": "pmRA",
            "pmra_error": "e_pmRA",
            "b": "GLAT",
            "pmdec": "pmDE",
            "pmdec_error": "e_pmDE",
            "l": "GLON",
            "phot_g_mean_flux": "FG",
            "phot_g_mean_flux_error": "e_FG",
            "phot_bp_mean_flux": "FBP",
            "phot_bp_mean_flux_error": "e_FBP",
            "phot_rp_mean_flux": "FRP",
            "phot_rp_mean_flux_error": "e_FRP",
            "radial_velocity": "RV",
            "radial_velocity_error": "e_RV",
        }
    )

    return all_frames


def uncertMags(
    Zp_G: float,
    Zp_BP: float,
    Zp_RP: float,
    sigma_ZG_2: float,
    sigma_ZBP_2: float,
    sigma_ZRP_2: float,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates magnitudes and uncertainties in the G, BP, and RP bands.

    Gaia DR3 zero points: https://www.cosmos.esa.int/web/gaia/dr3-passbands
    "The GBP (blue curve), G (green curve) and GRP (red curve) passbands are
    applicable to both Gaia Early Data Release 3 as well as to the full Gaia
    Data Release 3"

    Parameters
    ----------
    Zp_G : float
        Zero point for the G band.
    Zp_BP : float
        Zero point for the BP band.
    Zp_RP : float
        Zero point for the RP band.
    sigma_ZG_2 : float
        Variance of the zero point for the G band.
    sigma_ZBP_2 : float
        Variance of the zero point for the BP band.
    sigma_ZRP_2 : float
        Variance of the zero point for the RP band.
    data : pd.DataFrame
        DataFrame containing Gaia data with flux measurements.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns for Gmag, BP-RP, e_Gmag, and e_BP-RP.
    """
    I_G, e_IG = np.array(data["FG"]), np.array(data["e_FG"])
    I_BP, e_IBP = np.array(data["FBP"]), np.array(data["e_FBP"])
    I_RP, e_IRP = np.array(data["FRP"]), np.array(data["e_FRP"])

    data["Gmag"] = Zp_G + -2.5 * np.log10(I_G)
    BPmag = Zp_BP + -2.5 * np.log10(I_BP)
    RPmag = Zp_RP + -2.5 * np.log10(I_RP)
    data["BP-RP"] = BPmag - RPmag

    e_G = np.sqrt(sigma_ZG_2 + 1.179 * (e_IG / I_G) ** 2)
    data["e_Gmag"] = e_G
    e_BP = np.sqrt(sigma_ZBP_2 + 1.179 * (e_IBP / I_BP) ** 2)
    e_RP = np.sqrt(sigma_ZRP_2 + 1.179 * (e_IRP / I_RP) ** 2)
    data["e_BP-RP"] = np.sqrt(e_BP**2 + e_RP**2)

    return data
