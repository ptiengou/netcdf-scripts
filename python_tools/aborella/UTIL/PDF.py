import numpy as np
import scipy.special as spe
import scipy.stats as stats

def diracPDF(x, mu):

    i_nonzero = np.argmin(np.abs(x - mu))
    diracPDF = np.zeros_like(x)
    diracPDF[i_nonzero] = 1.

    return diracPDF


def diracCDF(x, mu):

    i_nonzero = np.argmin(np.abs(x - mu))
    diracPDF = np.zeros_like(x)
    x[i_nonzero:] = 1.

    return diracCDF


def gaussianPDF(x, mu, sigma):

    tmp1 = np.power(x - mu, 2.) / (2. * (sigma**2.))
    gaussianPDF = np.exp( - tmp1 ) / (sigma * np.sqrt( 2.*np.pi ))

    return gaussianPDF


def gaussianCDF(x, mu, sigma):

    tmp1 = spe.erf( (x - mu) / (sigma * np.sqrt(2.)) )
    gaussianCDF = (1. + tmp1) / 2.

    return gaussianCDF


def generalisedNormalMeanStdPDF(x, mu, sigma, borneX=0.):

    if mu == borneX: raise ValueError('The mean mu and the limit borneX must be different.')

    kappa = - np.sqrt( np.log(1. + np.power(sigma / (mu - borneX), 2.)) )
    if kappa != 0.:
        alpha = - kappa * (mu - borneX) * np.exp( - (kappa**2.) / 2.)
    else:
        return
    lambd = borneX + (mu - borneX) * np.exp( - (kappa**2.) / 2.)

    generalisedNormalMomentsPDF = generalisedNormalParamPDF(x, lambd, alpha, kappa)

    return generalisedNormalMomentsPDF


def generalisedNormalMeanStdCDF(x, mu, sigma, borneX=0.):

    if mu == borneX: raise ValueError('The mean mu and the limit borneX must be different.')

    kappa = - np.sqrt( np.log(1. + np.power(sigma / (mu - borneX), 2.)) )
    if kappa != 0.:
        alpha = - kappa * (mu - borneX) * np.exp( - (kappa**2.) / 2.)
    else:
        return
    lambd = borneX + (mu - borneX) * np.exp( - (kappa**2.) / 2.)

    generalisedNormalMomentsCDF = generalisedNormalParamCDF(x, lambd, alpha, kappa)

    return generalisedNormalMomentsCDF


def generalisedNormalParamPDF(x, lambd, alpha, kappa):
    """
    lambda: position parameter
    alpha: scale parameter
    kappa: shape parameter
    """

    if alpha == 0.: raise ValueError('The scale parameter must be different from 0.')

    #if kappa > 0.:
    #    if (x > lambd + alpha/kappa).any(): raise ValueError('The values of x are off bound for the given parameters.')
    #elif kappa < 0.:
    #    if (x < lambd + alpha/kappa).any(): raise ValueError('The values of x are off bound for the given parameters.')


    if kappa > 0.:
        i_valid = np.nonzero( x < lambd + alpha/kappa )
    elif kappa < 0.:
        i_valid = np.nonzero( x > lambd + alpha/kappa )
    else:
        i_valid = np.full( len(x), fill_value=True )
    generalisedNormalParamPDF = np.zeros_like(x)


    if kappa == 0.:
        y = (x[i_valid] - lambd) / alpha
    else:
        tmp1 = 1. - kappa * (x[i_valid] - lambd) / alpha
        y = - np.log(tmp1) / kappa

    tmp2 = np.exp(kappa * y - np.power(y, 2.) / 2.)
    generalisedNormalParamPDF[i_valid] = tmp2 / (alpha * np.sqrt( 2.*np.pi ))

    return generalisedNormalParamPDF


def generalisedNormalParamCDF(x, lambd, alpha, kappa):
    """
    lambda: position parameter
    alpha: scale parameter
    kappa: shape parameter
    """

    if alpha == 0.: raise ValueError('The scale parameter must be different from 0.')

    if kappa > 0.:
        if (x > lambd + alpha/kappa).any(): raise ValueError('The values of x are off bound for the given parameters.')
    elif kappa < 0.:
        if (x < lambd + alpha/kappa).any(): raise ValueError('The values of x are off bound for the given parameters.')
    

    if kappa == 0.:
        y = (x - lambd) / alpha
    else:
        tmp1 = 1. - kappa * (x - lambd) / alpha
        y = - np.log(tmp1) / kappa

    generalisedNormalParamCDF = gaussianCDF(y, 0., 1.)

    return generalisedNormalParamCDF


def gammaMeanStdPDF(x, mu, sigma, borneX=0.):

    #if (x - borneX <= 0.).any(): raise ValueError('The values of x are off bound for the given parameters.')

    alpha = (mu**2.) / (sigma**2.)
    beta = mu / (sigma**2.)

    gammaMomentsPDF = stats.gamma.pdf(x, alpha, borneX, 1./beta)#gammaParamPDF(x, alpha, beta, lambd=borneX)
    gamma_func = stats.gamma(alpha, borneX, 1./beta)

    return gamma_func#gammaMomentsPDF


def gammaMeanStdCDF(x, mu, sigma, borneX=0.):

    if (x - borneX <= 0.).any(): raise ValueError('The values of x are off bound for the given parameters.')

    alpha = (mu**2.) / (sigma**2.)
    beta = mu / (sigma**2.)

    gammaMomentsCDF = gammaParamCDF(x, alpha, beta, lambd=borneX)

    return gammaMomentsCDF


def gammaParamPDF(x, alpha, beta, lambd=0.):
    """
    lambda: position parameter
    alpha: shape parameter
    beta: intensity parameter
    """

    if (x - lambd <= 0.).any(): raise ValueError('The values of x are off bound for the given parameters.')

    tmp1 = np.power(x - lambd, alpha - 1.) * np.exp( - beta * (x - lambd) )
    gammaParamPDF = tmp1 * np.power(beta, alpha) / spe.gamma(alpha)

    return gammaParamPDF


def gammaParamCDF(x, alpha, beta, lambd=0.):
    """
    lambda: position parameter
    alpha: shape parameter
    beta: intensity parameter
    """

    if (x - lambd <= 0.).any(): raise ValueError('The values of x are off bound for the given parameters.')

    gammaParamCDF = spe.gammainc(alpha, (x - lambd) * beta)

    return gammaParamCDF


def betaMeanStdPDF(x, mu, sigma, x_bounds=(0.,1.)):

    if type(x_bounds) != tuple or len(x_bounds) != 2: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    x_min, x_max = x_bounds
    if x_min >= x_max: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    tmp = ( (x_max - mu) * (mu - x_min) / (sigma)**2. - 1. ) / (x_max - x_min)
    alpha = tmp * (mu - x_min)
    beta = tmp * (x_max - mu)
    #print(alpha, beta)
    #betaMeanStdPDF = betaParamPDF(x, alpha, beta, x_bounds)
    betaMeanStdPDF = stats.beta.pdf(x, alpha, beta, loc=x_min, scale=(x_max-x_min))
    beta_func = stats.beta(alpha, beta, loc=x_min, scale=(x_max-x_min))

    return beta_func#betaMeanStdPDF


def betaMeanStdCDF(x, mu, sigma, x_bounds=(0.,1.)):

    if type(x_bounds) != tuple or len(x_bounds) != 2: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    x_min, x_max = x_bounds
    if x_min >= x_max: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    tmp = ( (x_max - mu) * (mu - x_min) / (sigma)**2. - 1. ) / (x_max - x_min)
    alpha = tmp * (mu - x_min)
    beta = tmp * (x_max - mu)

    betaMeanStdCDF = betaParamCDF(x, alpha, beta, x_bounds)

    return betaMeanStdCDF


def betaMeanSkewPDF(x, mu, gamma, x_bounds=(0.,1.)):

    if type(x_bounds) != tuple or len(x_bounds) != 2: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    x_min, x_max = x_bounds
    if x_min >= x_max: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    mu_p = (mu - x_min) / (x_max - x_min)
    gamma_p = gamma

    tmp = (1. - 2. * mu_p)**2. / (1. - mu_p)
    max_val = tmp / mu_p / np.sqrt(3.)
    if (gamma_p)**2. > max_val : raise ValueError('Maximum value for the squared skewness exceeded. ' + \
                                                  'The max value allowed is %.4f, but '%(max_val) + \
                                                  'the value of the squared skewness is %.4f.'%((gamma_p)**2.))
   
    tmp = 2. * tmp / (gamma_p)**2.
    alpha = tmp - mu_p + np.sqrt( (tmp)**2. - 3. * (mu_p)**2. )
    beta = alpha * ( 1. / mu_p - 1. )

    betaMeanSkewPDF = betaParamPDF(x, alpha, beta, x_bounds)

    return betaMeanSkewPDF


def betaParamPDF(x, alpha, beta, x_bounds=(0.,1.)):

    if type(x_bounds) != tuple or len(x_bounds) != 2: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    x_min, x_max = x_bounds
    if x_min >= x_max: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    #if (x < x_min).any() or (x > x_max).any(): raise ValueError('The values of x are off bound for the given parameters.')

    x_remap = np.interp(x, (x_min, x_max), (0., 1.))
    i_valid = np.nonzero( np.logical_and(x_remap > 0., x_remap < 1.) )
    betaParamPDF = np.zeros_like(x_remap)

    tmp1 = np.power(x_remap[i_valid], alpha - 1.) * np.power(1. - x_remap[i_valid], beta - 1.)
    betaParamPDF[i_valid] = tmp1 / spe.beta(alpha, beta) / ( x_max - x_min )

    return betaParamPDF


def betaParamCDF(x, alpha, beta, x_bounds=(0.,1.)):

    if type(x_bounds) != tuple or len(x_bounds) != 2: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    x_min, x_max = x_bounds
    if x_min >= x_max: raise ValueError('x_bounds must be a tuple (x_min, x_max) with x_min < x_max.')

    x_remap = np.interp(x, (x_min, x_max), (0., 1.))
    i_valid = np.nonzero( np.logical_and(x_remap > 0., x_remap < 1.) )
    betaParamCDF = np.zeros_like(x_remap)

    betaParamCDF[i_valid] = spe.betainc(alpha, beta, x_remap[i_valid])

    return betaParamCDF


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    
    x = np.linspace(0.,2.2,1000)
    plt.plot(x, generalisedNormalMeanStdPDF(x, 1.2, 0.5, borneX=0.))
    plt.show()
