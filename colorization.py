

import numpy as np
import cv2
from typing import Tuple


# ============================================================
# CONSTANTES DEL PAPER ORIGINAL (Giacomelli et al., 2020)
# ============================================================
HE_COLOR_SETTINGS = {
    'nuclei': [0.17, 0.27, 0.105],   # R, G, B hematoxilina
    'cyto':   [0.05, 1.0,  0.54],    # R, G, B eosina
}

BETA_DICT = {
    'K_nuclei': 0.08,    # factor multiplicativo nuclear
    'K_cyto':   0.012,   # factor multiplicativo citoplasma
}


# ============================================================
# FUNCIONES DEL CÓDIGO ORIGINAL (adaptadas CPU-only)
# ============================================================

def getBackgroundLevels(image, threshold=50):
    """
    Calcula niveles de fondo y primer plano.
    Original: coloring.py → getBackgroundLevels()
    """
    image_sorted = np.sort(image, axis=None)
    foreground_vals = image_sorted[image_sorted > threshold]

    if len(foreground_vals) == 0:
        return threshold, threshold / 5

    hi_val = foreground_vals[int(np.round(len(foreground_vals) * 0.95))]
    background = hi_val / 5

    return hi_val, background


def preProcess(image, threshold=50, normfactor=None):
    """
    Preprocesamiento original de FalseColor-Python.
    Original: coloring.py → preProcess()

    1. Resta background
    2. Elimina negativos
    3. Potencia 0.85
    4. Normaliza a rango 8-bit
    """
    image = image.astype(float)

    # Resta de fondo
    image -= threshold

    # Sin valores negativos
    image[image < 0] = 0

    # Potencia 0.85 (compresión de rango dinámico)
    image = np.power(image, 0.85)

    # Factor de normalización
    if normfactor is None:
        foreground = image[image > threshold]
        if len(foreground) > 0:
            normfactor = np.mean(foreground) * 8
        else:
            normfactor = 1

    # Convertir a rango 8-bit
    processed_image = image * (65535 / normfactor) * (255 / 65535)

    return processed_image


def falseColor(nuclei, cyto,
               nuc_threshold=50,
               cyto_threshold=50,
               nuc_normfactor=5000,
               cyto_normfactor=2000):
    """
    Colorización FalseColor H&E virtual — CÓDIGO ORIGINAL.
    Original: coloring.py → falseColor()

    Usa Beer's Law con constantes empíricas del paper:
      RGB[i] = 255 * exp(-(nuc_settings[i] * K_nuc * nuclei_pp +
                            cyto_settings[i] * K_cyto * cyto_pp))

    Parameters
    ----------
    nuclei : 2D numpy array
        Canal nuclear (hematoxilina equivalente)
    cyto : 2D numpy array
        Canal citoplasma (eosina equivalente)
    nuc_threshold : int
        Nivel de fondo para canal nuclear
    cyto_threshold : int
        Nivel de fondo para canal citoplasma
    nuc_normfactor : int
        Factor de normalización nuclear (mayor = menos saturado)
    cyto_normfactor : int
        Factor de normalización citoplasma

    Returns
    -------
    RGB_image : 3D numpy array uint8, shape [H, W, 3]
    """
    constants_nuclei = HE_COLOR_SETTINGS['nuclei']
    k_nuclei = BETA_DICT['K_nuclei']

    constants_cyto = HE_COLOR_SETTINGS['cyto']
    k_cytoplasm = BETA_DICT['K_cyto']

    # Preprocesamiento (idéntico al original)
    nuclei = nuclei.astype(float)
    nuclei = preProcess(nuclei, threshold=nuc_threshold,
                        normfactor=nuc_normfactor)

    cyto = cyto.astype(float)
    cyto = preProcess(cyto, threshold=cyto_threshold,
                      normfactor=cyto_normfactor)

    # Crear array RGB
    RGB_image = np.zeros((3, nuclei.shape[0], nuclei.shape[1]))

    # Beer-Lambert por canal
    for i in range(3):
        tmp_c = constants_cyto[i] * k_cytoplasm * cyto
        tmp_n = constants_nuclei[i] * k_nuclei * nuclei
        RGB_image[i] = 255 * np.multiply(np.exp(-tmp_c), np.exp(-tmp_n))

    # Reordenar a [H, W, C]
    RGB_image = np.moveaxis(RGB_image, 0, -1)

    return RGB_image.astype(np.uint8)


def applyCLAHE(image, clip_limit=0.048, tile_size=(8, 8)):
    """
    CLAHE del código original.
    Original: coloring.py → applyCLAHE()
    """
    input_dtype = np.uint16
    clahe = cv2.createCLAHE(
        tileGridSize=tile_size,
        clipLimit=clip_limit
    )
    image_16 = image.astype(input_dtype)
    equalized = clahe.apply(image_16)

    # Renormalizar a niveles originales
    if equalized.max() > 0:
        final = (image_16.max()) * (equalized / equalized.max())
    else:
        final = equalized

    return final.astype(input_dtype)


# ============================================================
# CLASE WRAPPER PARA INTEGRACIÓN CON APP.PY
# ============================================================

class HistologyColorizer:
    """
    Wrapper que integra el FalseColor-Python original
    con el pipeline de la app Streamlit.
    """

    @staticmethod
    def split_otls_channels(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae los dos canales de fluorescencia de una imagen OTLS.
        Auto-detecta por varianza cuál canal es núcleos y cuál citoplasma.
        """
        if img_rgb.ndim == 2:
            return img_rgb.copy(), img_rgb.copy()

        channels = [img_rgb[:, :, i] for i in range(3)]
        stds = [float(np.std(ch)) for ch in channels]
        sorted_idx = np.argsort(stds)[::-1]

        # Mayor varianza → señal nuclear (distribución bimodal)
        ch_nuclei = channels[sorted_idx[0]]
        # Segunda varianza → citoplasma
        ch_cyto = channels[sorted_idx[1]]

        return ch_nuclei.copy(), ch_cyto.copy()

    @staticmethod
    def colorize(img: np.ndarray,
                 nuc_threshold: int = 50,
                 cyto_threshold: int = 50,
                 nuc_normfactor: int = 5000,
                 cyto_normfactor: int = 2000,
                 use_clahe: bool = False,
                 clahe_clip: float = 0.048) -> np.ndarray:
        """
        Pipeline completo: imagen OTLS → FalseColor H&E.

        Usa la función falseColor() original del paper.

        Parameters
        ----------
        img : numpy array (H,W) o (H,W,3)
            Imagen OTLS de fluorescencia
        nuc_threshold : int
            Umbral de fondo canal nuclear
        cyto_threshold : int
            Umbral de fondo canal citoplasma
        nuc_normfactor : int
            Normalización nuclear (mayor = menos saturado)
        cyto_normfactor : int
            Normalización citoplasma
        use_clahe : bool
            Aplicar CLAHE antes de colorizar
        clahe_clip : float
            Clip limit para CLAHE

        Returns
        -------
        RGB image uint8 shape [H, W, 3]
        """
        # Separar canales
        if img.ndim == 2:
            nuclei = img.copy()
            cyto = img.copy()
        else:
            nuclei, cyto = HistologyColorizer.split_otls_channels(img)

        # CLAHE opcional
        if use_clahe:
            nuclei = applyCLAHE(
                nuclei.astype(np.uint16),
                clip_limit=clahe_clip
            ).astype(np.uint16)
            cyto = applyCLAHE(
                cyto.astype(np.uint16),
                clip_limit=clahe_clip
            ).astype(np.uint16)

        # Aplicar FalseColor original
        result = falseColor(
            nuclei, cyto,
            nuc_threshold=nuc_threshold,
            cyto_threshold=cyto_threshold,
            nuc_normfactor=nuc_normfactor,
            cyto_normfactor=cyto_normfactor,
        )

        return result


class SuperResolution:
    """Super-resolución Lanczos + sharpening."""

    @staticmethod
    def apply(img: np.ndarray,
              scale: int = 2,
              sharpen_strength: float = 0.4) -> np.ndarray:
        if scale not in (2, 4):
            raise ValueError("scale debe ser 2 o 4")

        h, w = img.shape[:2]
        upscaled = cv2.resize(
            img, (w * scale, h * scale),
            interpolation=cv2.INTER_LANCZOS4
        )

        if sharpen_strength > 0:
            blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2.0)
            upscaled = cv2.addWeighted(
                upscaled, 1.0 + sharpen_strength,
                blurred, -sharpen_strength, 0
            )
            upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

        return upscaled