import numpy as np
import matplotlib.pyplot as plt

M_eau = 18.01528 * 1e-3 # kg/mol
M_air = 28.9645 * 1e-3 # kg/mol
eps = M_eau / M_air

RV = 1000.0 * 1.380658E-23 * 6.0221367E+23 / 18.0153
RD = 1000.0 * 1.380658E-23 * 6.0221367E+23 / 28.9644
RLSTT = 2.834e6
EPS_W = eps


def RHiToRHw(RHi, t, equation='Sonntag'):
    p_sat_ice = getSatPressure(t, phase='ice', equation=equation)
    p_sat_liq = getSatPressure(t, phase='liq', equation=equation)
    RHw = RHi / p_sat_liq * p_sat_ice
    return RHw


def RHwToRHi(RHw, t, equation='Sonntag'):
    p_sat_ice = getSatPressure(t, phase='ice', equation=equation)
    p_sat_liq = getSatPressure(t, phase='liq', equation=equation)
    RHi = RHw * p_sat_liq / p_sat_ice
    return RHi


def volMixRatioToRH(x, t, p, **kwargs):
    # x: H2O in mol/mol
    # t: temperature in K
    # p: pres in Pa
    # RH: relative humidity in %
    x_sat = getSatVolMixRatio(t, p, **kwargs)
    RH = x / x_sat * 1e2 # in %
    return RH


def RHtoVolMixRatio(RH, t, p, **kwargs):
    # RH: relative humidity in %
    # t: temperature in K
    # p: pres in Pa
    # x: H2O in mol/mol
    x_sat = getSatVolMixRatio(t, p, **kwargs)
    x = ( RH / 1e2 ) * x_sat # in mol/mol
    return x


def massMixRatioToRH(w, t, p, **kwargs):
    # w: H2O in kg/kg(dry air)
    # t: temperature in K
    # p: pres in Pa
    # RH: relative humidity in %
#    w_sat = getSatMassMixRatio(t, p, **kwargs)
    x = massMixRatioToVolMixRatio(w)
    RH = volMixRatioToRH(x, t, p, **kwargs)
    return RH


def RHtoMassMixRatio(RH, t, p, **kwargs):
    # RH: relative humidity in %
    # t: temperature in K
    # p: pres in Pa
    # w: H2O in kg/kg(dry air)
#    w_sat = getSatVolMixRatio(t, p, **kwargs)
    x = RHtoVolMixRatio(RH, t, p, **kwargs)
    w = volMixRatioToMassMixRatio(x)
    return w


def speHumToRH(q, t, p, **kwargs):
    # q: H2O in kg/kg(wet air)
    # t: temperature in K
    # p: pres in Pa
    # RH: relative humidity in %
#    q_sat = getSatSpeHum(t, p, **kwargs)
    x = speHumToVolMixRatio(q)
    RH = volMixRatioToRH(x, t, p, **kwargs)
    return RH


def RHtoSpeHum(RH, t, p, **kwargs):
    # RH: relative humidity in %
    # t: temperature in K
    # p: pres in Pa
    # q: H2O in kg/kg(wet air)
#    q_sat = getSatSpeHum(t, p, **kwargs)
    x = RHtoVolMixRatio(RH, t, p, **kwargs)
    q = volMixRatioToSpeHum(x)
    return q


def getSatVolMixRatio(t, p, **kwargs):
    p_sat = getSatPressure(t, **kwargs)
    x_sat = p_sat / p
    return x_sat


def getSatMassMixRatio(t, p, **kwargs):
    p_sat = getSatPressure(t, **kwargs)
    w_sat = eps * p_sat / ( p - p_sat )
    return w_sat


def getSatSpeHum(t, p, **kwargs):
    p_sat = getSatPressure(t, **kwargs)
    q_sat = eps * p_sat / ( p - ( 1. - eps ) * p_sat )
    return q_sat


def getSatPressure(t, phase='ice', equation='Sonntag'):
    # from temp in K to p_sat in Pa

    if equation == 'Sonntag':
        if phase == 'ice':
            a = - 6024.5282
            b = 29.32707
            c = 1.0613868e-2
            d = - 1.3198825e-5
            e = - 0.49382577

        else:
            a = - 6096.9385
            b = 21.2409642
            c = - 2.711193e-2
            d = 1.673952e-5
            e = 2.433502

        p_sat = a / t + b + c * t + d * t**2. + e * np.log(t)
        p_sat = np.exp( p_sat )

    elif equation == 'Hardy':
        if phase == 'ice':
            a = 0.
            b = - 5.8666426e3
            c = 2.232870244e1
            d = 1.39387003e-2
            e = - 3.4262402e-5
            f = 2.7040955e-8
            g = 0.
            h = 6.7063522e-1

        else:
            a = - 2.8365744e3
            b = - 6.028076559e3
            c = 1.954263612e1
            d = - 2.737830188e-2
            e = 1.6261698e-5
            f = 7.0229056e-10
            g = - 1.8680009e-13
            h = 2.7150305

        p_sat = a / t**2. + b / t + c + d * t + e * t**2. \
              + f * t**3. + g * t**4. + h * np.log(t)
        p_sat = np.exp( p_sat )

    elif equation == 'Buck':
        T_0 = 273.15

        if phase == 'ice':
            a = 6.1115
            b = 23.036
            c = 333.7
            d = 279.82
            tmp_1 = ( 23.036 - ( t - 273.15 ) / 333.7 )
            tmp_2 = ( t - 273.15 ) / ( 279.82 + t - 273.15 )
            p_sat = 6.1115 * np.exp( tmp_1 * tmp_2 )

        else:
            a = 6.1121
            b = 18.678
            c = 234.5
            d = 257.14

        p_sat = (b - (t - T_0) / c) * (t - T_0) / (d + t - T_0)
        p_sat = a * np.exp( p_sat ) * 1e2

    else:
        raise ValueError(f'{equation} not recognised, '
                          'only the Sonntag, Hardy and Buck equations are implemented.')

    return p_sat


def volMixRatioToMassMixRatio(x):
    w = eps * x / ( 1. - x )
    return w


def volMixRatioToSpeHum(x):
    q = eps * x / ( 1. - ( 1. - eps ) * x )
    return q


def massMixRatioToVolMixRatio(w):
    x = w / ( w + eps )
    return x


def massMixRatioToSpeHum(w):
    q = w / ( 1. + w )
    return q


def speHumToVolMixRatio(q):
    x = q / ( eps + ( 1. - eps ) * q )
    return x


def speHumToMassMixRatio(q):
    w = q / ( 1. - q )
    return w


def getHomoThreshold(t):
    # formula from Kärcher and Burkhardt 2008
    gamma_ss = 2.349 - t / 259.
    return gamma_ss


def getCondThreshold(t):
    homo_thresh = getHomoThreshold(t)
    esat_liq = getSatPressure(t, phase='liq')
    esat_ice = getSatPressure(t, phase='ice')
    if ( homo_thresh < esat_liq / esat_ice ):
        gamma = homo_thresh
    else:
        gamma = esat_liq / esat_ice
    return gamma


def convertWater(origin_var, destination_var, w_val, t_val, p_val, **kwargs):

    if origin_var == 'q':
        q_val = w_val
    elif origin_var == 'sl':
        q_val = w_val + getSatSpeHum(t_val, p_val, phase='liq', **kwargs)
    elif origin_var == 'si':
        q_val = w_val + getSatSpeHum(t_val, p_val, phase='ice', **kwargs)
    elif origin_var == 'rl':
        q_val = RHtoSpeHum(w_val, t_val, p_val, phase='liq', **kwargs)
    elif origin_var == 'ri':
        q_val = RHtoSpeHum(w_val, t_val, p_val, phase='ice', **kwargs)
    elif origin_var == 'xv':
        q_val = volMixRatioToSpeHum(w_val)
    elif origin_var == 'w':
        q_val = massMixRatioToSpeHum(w_val)


    if destination_var == 'q':
        w_final = q_val
    elif destination_var == 'sl':
        w_final = q_val - getSatSpeHum(t_val, p_val, phase='liq', **kwargs)
    elif destination_var == 'si':
        w_final = q_val - getSatSpeHum(t_val, p_val, phase='ice', **kwargs)
    elif destination_var == 'rl':
        w_final = speHumToRH(q_val, t_val, p_val, phase='liq', **kwargs)
    elif destination_var == 'ri':
        w_final = speHumToRH(q_val, t_val, p_val, phase='ice', **kwargs)
    elif destination_var == 'xv':
        w_final = speHumToVolMixRatio(q_val)
    elif destination_var == 'w':
        w_final = speHumToMassMixRatio(q_val)


    return w_final
