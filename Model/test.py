def daylenght(doy,L):
    from math import tan,atan,sin,asin,cos,acos,pi
    theta=0.2163108+2*atan(0.9671396*tan(0.00860*(doy-186)))
    phi=asin(0.39795*cos(theta))

    D= 24- (24/pi)*acos((sin((L*pi))/(180)*sin(phi))/(cos((L*pi)/180)*cos(phi)))

    return D

print(daylenght(150,10.5))