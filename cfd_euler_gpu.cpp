#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

// Global parameters
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

// Compute pressure from the conservative variables
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// Compute flux in the x-direction
void fluxX(double rho, double rhou, double rhov, double E, 
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// Compute flux in the y-direction
void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

int main() {
    // Grid and domain parameters
    const int Nx = 200;         // Number of cells in x (excluding ghost cells)
    const int Ny = 100;         // Number of cells in y
    const double Lx = 2.0;      // Domain length in x
    const double Ly = 1.0;      // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    auto t1 = chrono::high_resolution_clock::now();

    // Create flat arrays (with ghost cells)
    const int total_size = (Nx + 2) * (Ny + 2);
    vector<double> rho(total_size), rhou(total_size), rhov(total_size), E(total_size);
    vector<double> rho_new(total_size), rhou_new(total_size), rhov_new(total_size), E_new(total_size);
    vector<bool> solid(total_size, false);

    // Obstacle (cylinder) parameters
    const double cx = 0.5;      // Cylinder center x
    const double cy = 0.5;      // Cylinder center y
    const double radius = 0.1;  // Cylinder radius

    // Free-stream initial conditions (inflow)
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0/(gamma_val - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);

    // Initialize grid and obstacle mask
    #pragma omp target teams distribute parallel for collapse(2) map(to:cx,cy,radius,rho0,u0,v0,p0,E0) map(tofrom:rho[:total_size],rhou[:total_size],rhov[:total_size],E[:total_size],solid[:total_size])
    for (int i = 0; i < Nx+2; i++) {
        for (int j = 0; j < Ny+2; j++) {
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            if ((x - cx)*(x - cx) + (y - cy)*(y - cy) <= radius * radius) {
                solid[i*(Ny+2)+j] = true;
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = 0.0;
                rhov[i*(Ny+2)+j] = 0.0;
                E[i*(Ny+2)+j] = p0/(gamma_val - 1.0);
            } else {
                solid[i*(Ny+2)+j] = false;
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = rho0 * u0;
                rhov[i*(Ny+2)+j] = rho0 * v0;
                E[i*(Ny+2)+j] = E0;
            }
        }
    }

    // Determine time step from CFL condition
    double c0 = sqrt(gamma_val * p0 / rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0) / 2.0;

    // Time stepping parameters
    const int nSteps = 2000;

    // Main time-stepping loop
    #pragma omp target data map(tofrom:rho[:total_size],rhou[:total_size],rhov[:total_size],E[:total_size]) \
                          map(alloc:rho_new[:total_size],rhou_new[:total_size],rhov_new[:total_size],E_new[:total_size])
    {
        for (int n = 0; n < nSteps; n++) {
            // Apply boundary conditions
            #pragma omp target teams distribute parallel for
            for (int j = 0; j < Ny+2; j++) {
                // Left boundary (inflow)
                rho[0*(Ny+2)+j] = rho0;
                rhou[0*(Ny+2)+j] = rho0*u0;
                rhov[0*(Ny+2)+j] = rho0*v0;
                E[0*(Ny+2)+j] = E0;
                // Right boundary (outflow)
                rho[(Nx+1)*(Ny+2)+j] = rho[Nx*(Ny+2)+j];
                rhou[(Nx+1)*(Ny+2)+j] = rhou[Nx*(Ny+2)+j];
                rhov[(Nx+1)*(Ny+2)+j] = rhov[Nx*(Ny+2)+j];
                E[(Nx+1)*(Ny+2)+j] = E[Nx*(Ny+2)+j];
            }

            #pragma omp target teams distribute parallel for
            for (int i = 0; i < Nx+2; i++) {
                // Bottom boundary (reflective)
                rho[i*(Ny+2)+0] = rho[i*(Ny+2)+1];
                rhou[i*(Ny+2)+0] = rhou[i*(Ny+2)+1];
                rhov[i*(Ny+2)+0] = -rhov[i*(Ny+2)+1];
                E[i*(Ny+2)+0] = E[i*(Ny+2)+1];
                // Top boundary (reflective)
                rho[i*(Ny+2)+(Ny+1)] = rho[i*(Ny+2)+Ny];
                rhou[i*(Ny+2)+(Ny+1)] = rhou[i*(Ny+2)+Ny];
                rhov[i*(Ny+2)+(Ny+1)] = -rhov[i*(Ny+2)+Ny];
                E[i*(Ny+2)+(Ny+1)] = E[i*(Ny+2)+Ny];
            }

            // Update interior cells using a Lax-Friedrichs scheme
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    if (solid[i*(Ny+2)+j]) {
                        rho_new[i*(Ny+2)+j] = rho[i*(Ny+2)+j];
                        rhou_new[i*(Ny+2)+j] = rhou[i*(Ny+2)+j];
                        rhov_new[i*(Ny+2)+j] = rhov[i*(Ny+2)+j];
                        E_new[i*(Ny+2)+j] = E[i*(Ny+2)+j];
                    } else {
                        // Lax averaging and flux computation
                        double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                        double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                        double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                        double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                        fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j],
                              fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                        fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j],
                              fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                        fluxY(rho[i*(Ny+2)+(j+1)], rhou[i*(Ny+2)+(j+1)], rhov[i*(Ny+2)+(j+1)], E[i*(Ny+2)+(j+1)],
                              fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                        fluxY(rho[i*(Ny+2)+(j-1)], rhou[i*(Ny+2)+(j-1)], rhov[i*(Ny+2)+(j-1)], E[i*(Ny+2)+(j-1)],
                              fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                        double dtdx = dt / (2 * dx);
                        double dtdy = dt / (2 * dy);

                        rho_new[i*(Ny+2)+j] = 0.25 * (rho[(i+1)*(Ny+2)+j] + rho[(i-1)*(Ny+2)+j] + 
                                                      rho[i*(Ny+2)+(j+1)] + rho[i*(Ny+2)+(j-1)]) -
                                              dtdx * (fx_rho1 - fx_rho2) - dtdy * (fy_rho1 - fy_rho2);
                        rhou_new[i*(Ny+2)+j] = 0.25 * (rhou[(i+1)*(Ny+2)+j] + rhou[(i-1)*(Ny+2)+j] + 
                                                       rhou[i*(Ny+2)+(j+1)] + rhou[i*(Ny+2)+(j-1)]) -
                                               dtdx * (fx_rhou1 - fx_rhou2) - dtdy * (fy_rhou1 - fy_rhou2);
                        rhov_new[i*(Ny+2)+j] = 0.25 * (rhov[(i+1)*(Ny+2)+j] + rhov[(i-1)*(Ny+2)+j] + 
                                                       rhov[i*(Ny+2)+(j+1)] + rhov[i*(Ny+2)+(j-1)]) -
                                               dtdx * (fx_rhov1 - fx_rhov2) - dtdy * (fy_rhov1 - fy_rhov2);
                        E_new[i*(Ny+2)+j] = 0.25 * (E[(i+1)*(Ny+2)+j] + E[(i-1)*(Ny+2)+j] + 
                                                    E[i*(Ny+2)+(j+1)] + E[i*(Ny+2)+(j-1)]) -
                                            dtdx * (fx_E1 - fx_E2) - dtdy * (fy_E1 - fy_E2);
                    }
                }
            }

            // Copy updated values back
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    rho[i*(Ny+2)+j] = rho_new[i*(Ny+2)+j];
                    rhou[i*(Ny+2)+j] = rhou_new[i*(Ny+2)+j];
                    rhov[i*(Ny+2)+j] = rhov_new[i*(Ny+2)+j];
                    E[i*(Ny+2)+j] = E_new[i*(Ny+2)+j];
                }
            }

            // Calculate total kinetic energy (on CPU for simplicity)
            if (n % 50 == 0) {
                double total_kinetic = 0.0;
                #pragma omp target teams distribute parallel for collapse(2) map(tofrom:total_kinetic) reduction(+:total_kinetic)
                for (int i = 1; i <= Nx; i++) {
                    for (int j = 1; j <= Ny; j++) {
                        double u = rhou[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                        double v = rhov[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                        total_kinetic += 0.5 * rho[i*(Ny+2)+j] * (u * u + v * v);
                    }
                }
                cout << "Step " << n << " completed, total kinetic energy: " << total_kinetic << endl;
            }
        }
    }

    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms_double = t2 - t1;
    cout << "Simulation time: " << ms_double.count() << " ms" << endl;

    return 0;
}