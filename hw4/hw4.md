# Part 1

## a

SNR(I2) = E(S2)/E(N2) = E(S1/2)/E(N2) = {E(S1)/E(N1)}/4 = SNR(I1)/4

## b

E(I1) = E(S1) + 2sum(S1 * N1) + E(N1)
E(I2) = E(S2) + 2sum(S2 * N2) + E(N2)

E(I1) - E(I2) = {E(S1) - E(S2)} + {0-0} + {M * sigma^2 - M * sigma^2}

= E(S1) - E(S1)/4 

= 3 * E(S1) / 4

## c

Let I2hat = I2 * c

E(I2hat) = c^2 (E(S1)/4 + M * sigma^2)

we want this value to equal E(I1)

= E(I1) = E(S1) + M * sigma^2

Rearranging, we get 

c^2 = 4{E(S1)+Msigma^2}/{E(S1)+4Msigma^2}

c = 2sqrt({E(S1)+Msigma^2}/{E(S1)+4Msigma^2})

## d

RMSE(I1, S1) = sqrt(sum{N1^2}/M) = sigma

RMSE(I2hat, S1) = sqrt(sum{(I2hat-S1)^2}/M)

= sqrt(sum{(c * I2 - S1)^2}/M)

We essentially multiply the noise level by c as well while matching energy of the S1 - hence, the RMSE will also be around c * sigma.