// Exercise the production Clover implementation, without editing production.
#define main production_main_not_used
#include "../src/flowed_gluon_ope.cpp"
#undef main
int main() {
  Lattice lat(6,6,4,4);
  for (double flux : {0.02,0.01,0.005}) {
    std::vector<Mat3> u(4*lat.volume,eye3());
    for (size_t s=0;s<lat.volume;++s) {
      int x=s%lat.nx;
      Mat3 uy=eye3();
      uy(0,0)=std::exp(cf(0,flux*x/2));
      uy(1,1)=std::exp(cf(0,-flux*x/2));
      u[4*s+1]=uy;
    }
    // All four leaves are away from the discontinuity at the periodic seam.
    size_t s=2+lat.nx*(2+lat.ny*(1+lat.nz));
    Mat3 f=clover(u,lat,s,0,1);
    double expected=std::sin(flux/2);
    double norm=2.*trace(mul(f,f)).real()/(flux*flux);
    if (std::abs(f(0,0).real()-expected)>1e-7 ||
        std::abs(f(1,1).real()+expected)>1e-7 ||
        std::abs(norm-1.)>5e-4) return 1;
    std::cout<<"flux="<<flux<<" F00="<<f(0,0)
             <<" even_trace_over_flux_squared="<<norm<<" PASS\n";
  }
  return 0;
}
