#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using cf = std::complex<float>;
using cd = std::complex<double>;

struct Mat3 {
  cf v[9]{};
  cf &operator()(int r, int c) { return v[3 * r + c]; }
  const cf &operator()(int r, int c) const { return v[3 * r + c]; }
};
static_assert(sizeof(Mat3) == 9 * sizeof(cf));

static inline Mat3 zero3() { return Mat3{}; }
static inline Mat3 eye3() {
  Mat3 a{};
  a(0, 0) = a(1, 1) = a(2, 2) = cf(1.0f, 0.0f);
  return a;
}
static inline Mat3 add(const Mat3 &a, const Mat3 &b) {
  Mat3 c;
  for (int i = 0; i < 9; ++i) c.v[i] = a.v[i] + b.v[i];
  return c;
}
static inline Mat3 sub(const Mat3 &a, const Mat3 &b) {
  Mat3 c;
  for (int i = 0; i < 9; ++i) c.v[i] = a.v[i] - b.v[i];
  return c;
}
static inline Mat3 scale(const Mat3 &a, float x) {
  Mat3 c;
  for (int i = 0; i < 9; ++i) c.v[i] = x * a.v[i];
  return c;
}
static inline Mat3 scale(const Mat3 &a, cf x) {
  Mat3 c;
  for (int i = 0; i < 9; ++i) c.v[i] = x * a.v[i];
  return c;
}
static inline Mat3 dagger(const Mat3 &a) {
  Mat3 c;
  for (int r = 0; r < 3; ++r)
    for (int s = 0; s < 3; ++s) c(r, s) = std::conj(a(s, r));
  return c;
}
static inline Mat3 mul(const Mat3 &a, const Mat3 &b) {
  Mat3 c{};
  for (int r = 0; r < 3; ++r)
    for (int s = 0; s < 3; ++s) {
      cf x = 0.0f;
      for (int k = 0; k < 3; ++k) x += a(r, k) * b(k, s);
      c(r, s) = x;
    }
  return c;
}
static inline cf trace(const Mat3 &a) { return a(0, 0) + a(1, 1) + a(2, 2); }
static inline cf determinant(const Mat3 &a) {
  return a(0,0)*(a(1,1)*a(2,2)-a(1,2)*a(2,1))
       - a(0,1)*(a(1,0)*a(2,2)-a(1,2)*a(2,0))
       + a(0,2)*(a(1,0)*a(2,1)-a(1,1)*a(2,0));
}
static inline double norm2(const Mat3 &a) {
  double x = 0.0;
  for (const auto &z : a.v) x += std::norm(z);
  return x;
}
static inline Mat3 traceless(Mat3 a) {
  const cf tr = trace(a) / 3.0f;
  a(0,0) -= tr; a(1,1) -= tr; a(2,2) -= tr;
  return a;
}
static inline Mat3 antiherm_traceless(const Mat3 &a) {
  Mat3 q = scale(sub(a, dagger(a)), 0.5f);
  const cf tr = trace(q) / 3.0f;
  q(0,0) -= tr; q(1,1) -= tr; q(2,2) -= tr;
  return q;
}

// QUDA's Cayley-Hamilton exp(iQ), with a zero-matrix guard and clamped acos.
static inline Mat3 exp_antiherm(const Mat3 &a) {
  const cf minus_i(0.0f, -1.0f);
  const Mat3 q = scale(a, minus_i); // Hermitian Q=-iA, so exp(iQ)=exp(A).
  const Mat3 q2 = mul(q, q);
  const double c1 = 0.5 * static_cast<double>(trace(q2).real());
  if (!(c1 > 1.0e-18)) return eye3();
  double c0 = static_cast<double>(determinant(q).real());
  const double sqrt_c1_3 = std::sqrt(c1 / 3.0);
  const double c0max = 2.0 * (c1 / 3.0) * sqrt_c1_3;
  int parity = 0;
  if (c0 < 0.0) { c0 = -c0; parity = 1; }
  double ratio = c0max > 0.0 ? c0 / c0max : 1.0;
  ratio = std::max(-1.0, std::min(1.0, ratio));
  const double theta = std::acos(ratio);
  const double u = sqrt_c1_3 * std::cos(theta / 3.0);
  const double w = std::sqrt(c1) * std::sin(theta / 3.0);
  const double u2 = u * u, w2 = w * w;
  const double denom = 9.0 * u2 - w2;
  if (std::abs(denom) < 1.0e-20) {
    // This branch is only relevant infinitesimally close to zero.
    Mat3 a2 = mul(a, a), a3 = mul(a2, a), a4 = mul(a3, a);
    Mat3 out = add(eye3(), a);
    out = add(out, scale(a2, 0.5f));
    out = add(out, scale(a3, 1.0f / 6.0f));
    return add(out, scale(a4, 1.0f / 24.0f));
  }
  const cd eiu = std::exp(cd(0.0, u));
  const cd e2iu = eiu * eiu;
  const double cosw = std::cos(w);
  const double sincw = std::abs(w) < 0.05
      ? 1.0 - (w2/6.0)*(1.0-(w2/20.0)*(1.0-(w2/42.0)*(1.0-w2/72.0)))
      : std::sin(w) / w;
  const cd emiu=std::conj(eiu);
  cd f0 = ((u2-w2)*e2iu + 8.0*u2*cosw*emiu
           + cd(0.0, 2.0*u*(3.0*u2+w2)*sincw)*emiu) / denom;
  cd f1 = (2.0*u*e2iu - 2.0*u*cosw*emiu
           + cd(0.0, (3.0*u2-w2)*sincw)*emiu) / denom;
  cd f2 = (e2iu - cosw*emiu - cd(0.0, 3.0*u*sincw)*emiu) / denom;
  if (parity) {
    f0 = std::conj(f0);
    f1 = -std::conj(f1);
    f2 = std::conj(f2);
  }
  Mat3 out = scale(eye3(), cf(f0));
  out = add(out, scale(q, cf(f1)));
  out = add(out, scale(q2, cf(f2)));
  return out;
}

struct Lattice {
  int nx, ny, nz, nt;
  size_t volume;
  std::array<std::vector<uint32_t>, 4> plus, minus;
  Lattice(int x, int y, int z, int t) : nx(x), ny(y), nz(z), nt(t) {
    volume = static_cast<size_t>(nx) * ny * nz * nt;
    for (int mu = 0; mu < 4; ++mu) { plus[mu].resize(volume); minus[mu].resize(volume); }
    #pragma omp parallel for schedule(static)
    for (long long si = 0; si < static_cast<long long>(volume); ++si) {
      size_t s = static_cast<size_t>(si);
      int x0 = s % nx; size_t q = s / nx;
      int y0 = q % ny; q /= ny;
      int z0 = q % nz; int t0 = q / nz;
      auto idx = [&](int x1, int y1, int z1, int t1) -> uint32_t {
        return static_cast<uint32_t>((((static_cast<size_t>(t1)*nz+z1)*ny+y1)*nx+x1));
      };
      plus[0][s]=idx((x0+1)%nx,y0,z0,t0); minus[0][s]=idx((x0+nx-1)%nx,y0,z0,t0);
      plus[1][s]=idx(x0,(y0+1)%ny,z0,t0); minus[1][s]=idx(x0,(y0+ny-1)%ny,z0,t0);
      plus[2][s]=idx(x0,y0,(z0+1)%nz,t0); minus[2][s]=idx(x0,y0,(z0+nz-1)%nz,t0);
      plus[3][s]=idx(x0,y0,z0,(t0+1)%nt); minus[3][s]=idx(x0,y0,z0,(t0+nt-1)%nt);
    }
  }
  size_t spatial_volume() const { return static_cast<size_t>(nx)*ny*nz; }
};

static inline const Mat3 &link(const std::vector<Mat3> &u, size_t s, int mu) {
  return u[4*s+mu];
}
static inline Mat3 staple(const std::vector<Mat3> &u, const Lattice &lat, size_t s, int mu) {
  Mat3 sum{};
  const size_t spmu = lat.plus[mu][s];
  for (int nu = 0; nu < 4; ++nu) if (nu != mu) {
    const size_t spnu = lat.plus[nu][s];
    const size_t smnu = lat.minus[nu][s];
    const size_t smnu_pmu = lat.plus[mu][smnu];
    Mat3 fwd = mul(mul(link(u,s,nu), link(u,spnu,mu)), dagger(link(u,spmu,nu)));
    Mat3 bwd = mul(mul(dagger(link(u,smnu,nu)), link(u,smnu,mu)), link(u,smnu_pmu,nu));
    sum = add(sum, add(fwd,bwd));
  }
  return sum;
}

static void flow_stage(const std::vector<Mat3> &in, std::vector<Mat3> &out,
                       std::vector<Mat3> &temp, const Lattice &lat, float eps, int stage) {
  const size_t nlink = 4 * lat.volume;
  #pragma omp parallel for schedule(static)
  for (long long li = 0; li < static_cast<long long>(nlink); ++li) {
    const size_t l = static_cast<size_t>(li), s = l / 4;
    const int mu = static_cast<int>(l % 4);
    Mat3 z = mul(staple(in,lat,s,mu), dagger(in[l]));
    float coeff = eps;
    if (stage == 1) {
      temp[l] = z; coeff = eps / 4.0f;
    } else if (stage == 2) {
      z = sub(scale(z,8.0f/9.0f), scale(temp[l],17.0f/36.0f));
      temp[l] = z;
    } else {
      z = sub(scale(z,3.0f/4.0f), temp[l]);
    }
    Mat3 a = antiherm_traceless(scale(z, coeff));
    out[l] = mul(exp_antiherm(a), in[l]);
  }
}

static void wilson_flow(std::vector<Mat3> &u, const Lattice &lat, int steps, float eps) {
  std::vector<Mat3> work(u.size()), temp(u.size());
  for (int step = 0; step < steps; ++step) {
    flow_stage(u,work,temp,lat,eps,1);
    flow_stage(work,u,temp,lat,eps,2);
    flow_stage(u,work,temp,lat,eps,3);
    u.swap(work);
    if ((step+1)%10==0 || step+1==steps)
      std::cout << "flow_step=" << step+1 << "/" << steps << std::endl;
  }
}

static inline Mat3 oriented_link(const std::vector<Mat3> &u, const Lattice &lat,
                                 size_t &s, int signed_dir) {
  if (signed_dir > 0) {
    const int mu=signed_dir-1; Mat3 a=link(u,s,mu); s=lat.plus[mu][s]; return a;
  }
  const int mu=-signed_dir-1; s=lat.minus[mu][s]; return dagger(link(u,s,mu));
}
static Mat3 loop4(const std::vector<Mat3> &u, const Lattice &lat, size_t s,
                  int a, int b, int c, int d) {
  Mat3 p=oriented_link(u,lat,s,a);
  p=mul(p,oriented_link(u,lat,s,b));
  p=mul(p,oriented_link(u,lat,s,c));
  return mul(p,oriented_link(u,lat,s,d));
}
static Mat3 clover(const std::vector<Mat3> &u, const Lattice &lat, size_t s, int mu, int nu) {
  const int m=mu+1,n=nu+1;
  Mat3 q=loop4(u,lat,s,m,n,-m,-n);
  q=add(q,loop4(u,lat,s,n,-m,-n,m));
  q=add(q,loop4(u,lat,s,-m,-n,m,n));
  q=add(q,loop4(u,lat,s,-n,m,n,-m));
  return scale(sub(q,dagger(q)),cf(0.0f,-1.0f/8.0f));
}
static constexpr int PAIRS[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
static int pair_index(int mu,int nu) {
  if(mu>nu) std::swap(mu,nu);
  for(int i=0;i<6;++i) if(PAIRS[i][0]==mu&&PAIRS[i][1]==nu) return i;
  throw std::runtime_error("invalid Lorentz pair");
}
static Mat3 get_field(const std::array<std::vector<Mat3>,6> &f,size_t s,int mu,int nu) {
  if(mu==nu) return zero3();
  const float sign=mu<nu?1.0f:-1.0f;
  return scale(f[pair_index(mu,nu)][s],sign);
}
static int levi4(int a,int b,int c,int d) {
  int x[4]={a,b,c,d};
  for(int i=0;i<4;++i) for(int j=i+1;j<4;++j) if(x[i]==x[j]) return 0;
  int inv=0; for(int i=0;i<4;++i) for(int j=i+1;j<4;++j) inv+=(x[i]>x[j]);
  return inv%2?-1:1;
}
static Mat3 dual_field(const std::array<std::vector<Mat3>,6> &f,size_t s,int mu,int nu) {
  Mat3 out{};
  for(int r=0;r<4;++r) for(int q=0;q<4;++q) {
    int e=levi4(mu,nu,r,q); if(e) out=add(out,scale(get_field(f,s,r,q),0.5f*e));
  }
  return out;
}

static std::array<std::vector<Mat3>,6> make_fields(const std::vector<Mat3> &u,const Lattice &lat) {
  std::array<std::vector<Mat3>,6> f;
  for(auto &x:f) x.resize(lat.volume);
  #pragma omp parallel for schedule(static)
  for(long long si=0;si<static_cast<long long>(lat.volume);++si)
    for(int p=0;p<6;++p) f[p][si]=clover(u,lat,si,PAIRS[p][0],PAIRS[p][1]);
  return f;
}

// axes: projection(legacy,traceless), channel(unpol,helicity),
// z_orientation(+z,-z,even_sum,odd_difference),
// component(combined,Mtiti,Mijij,ti,tj,ij_single), z, t.
struct OpeData {
  int nz,nt;
  std::vector<cd> a;
  OpeData(int z,int t):nz(z),nt(t),a(static_cast<size_t>(2)*2*4*6*z*t){}
  cd &at(int p,int c,int o,int k,int z,int t) {
    size_t i=(((((static_cast<size_t>(p)*2+c)*4+o)*6+k)*nz+z)*nt+t);
    return a[i];
  }
};

static OpeData compute_ope(const std::vector<Mat3> &u,const Lattice &lat,
                           const std::array<std::vector<Mat3>,6> &f,int zdir,int zcount) {
  OpeData out(zcount,lat.nt);
  const int i=(zdir+1)%3,j=(zdir+2)%3;
  const int mus[3]={3,3,i}, nus[3]={i,j,j};
  std::vector<Mat3> w(lat.volume),wnext(lat.volume);
  for(int direction=0;direction<2;++direction) {
    #pragma omp parallel for schedule(static)
    for(long long s=0;s<static_cast<long long>(lat.volume);++s) w[s]=eye3();
    std::vector<uint32_t> endpoint(lat.volume);
    #pragma omp parallel for schedule(static)
    for(long long s=0;s<static_cast<long long>(lat.volume);++s) endpoint[s]=static_cast<uint32_t>(s);
    for(int z=0;z<zcount;++z) {
      if(z>0) {
        #pragma omp parallel for schedule(static)
        for(long long si=0;si<static_cast<long long>(lat.volume);++si) {
          size_t s=static_cast<size_t>(si);
          size_t e=endpoint[s];
          if(direction==0) { wnext[s]=mul(w[s],link(u,e,zdir)); endpoint[s]=lat.plus[zdir][e]; }
          else { size_t en=lat.minus[zdir][e]; wnext[s]=mul(w[s],dagger(link(u,en,zdir))); endpoint[s]=en; }
        }
        w.swap(wnext);
      }
      #pragma omp parallel for schedule(static)
      for(int t=0;t<lat.nt;++t) {
        cd sums[2][2][3]{};
        size_t begin=static_cast<size_t>(t)*lat.spatial_volume(), end=begin+lat.spatial_volume();
        for(size_t s=begin;s<end;++s) {
          const size_t e=endpoint[s]; const Mat3 wd=dagger(w[s]);
          for(int k=0;k<3;++k) {
            Mat3 fu1=get_field(f,e,mus[k],nus[k]);
            Mat3 fu2=get_field(f,s,mus[k],nus[k]);
            Mat3 fh1=fu1, fh2=dual_field(f,s,mus[k],nus[k]);
            for(int p=0;p<2;++p) {
              Mat3 a1=p?traceless(fu1):fu1, a2=p?traceless(fu2):fu2;
              Mat3 b1=p?traceless(fh1):fh1, b2=p?traceless(fh2):fh2;
              cf xu=trace(mul(a1,mul(wd,mul(a2,w[s]))));
              cf xh=trace(mul(b1,mul(wd,mul(b2,w[s]))));
              sums[p][0][k]+=cd(xu.real(),xu.imag()); sums[p][1][k]+=cd(xh.real(),xh.imag());
            }
          }
        }
        for(int p=0;p<2;++p) for(int c=0;c<2;++c) {
          const cd mtiti=sums[p][c][0]+sums[p][c][1];
          const cd mijij=2.0*sums[p][c][2];
          // Unpolarized: O = Mtiti-Mijij (arXiv:2510.26425v2).
          // Helicity: arXiv:2207.08733v2 Eq. (7) uses
          // Mtilde_0i;0i + Mtilde_ij;ij.  In our Euclidean index order the
          // time direction is 3, hence the direct stored combination is
          // Mtiti+Mijij.  Raw groups are retained independently as well.
          out.at(p,c,direction,0,z,t)=c?(mtiti+mijij):(mtiti-mijij);
          out.at(p,c,direction,1,z,t)=mtiti;
          out.at(p,c,direction,2,z,t)=mijij;
          out.at(p,c,direction,3,z,t)=sums[p][c][0];
          out.at(p,c,direction,4,z,t)=sums[p][c][1];
          out.at(p,c,direction,5,z,t)=sums[p][c][2];
        }
      }
    }
  }
  // arXiv:2207.08733v2 Eq. (2) defines the polarized matrix element as
  // m(z)-m(-z).  Save both raw orientations and explicit sum/difference for
  // every component; no reality projection or factor of 1/2 is applied.
  #pragma omp parallel for collapse(5) schedule(static)
  for(int p=0;p<2;++p) for(int c=0;c<2;++c) for(int k=0;k<6;++k)
    for(int z=0;z<zcount;++z) for(int t=0;t<lat.nt;++t) {
      const cd plus=out.at(p,c,0,k,z,t), minus=out.at(p,c,1,k,z,t);
      out.at(p,c,2,k,z,t)=plus+minus;
      out.at(p,c,3,k,z,t)=plus-minus;
    }
  return out;
}

static uint64_t be64(const unsigned char *p) {
  uint64_t x; std::memcpy(&x,p,8);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  x=__builtin_bswap64(x);
#endif
  return x;
}
static uint32_t be32(const unsigned char *p) {
  uint32_t x; std::memcpy(&x,p,4);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  x=__builtin_bswap32(x);
#endif
  return x;
}
struct LimePayload { size_t offset=0,bytes=0; };
static LimePayload find_lime_payload(const unsigned char *base,size_t n) {
  size_t pos=0;
  while(pos+144<=n) {
    if(be32(base+pos)!=0x456789abU) throw std::runtime_error("invalid LIME magic at offset "+std::to_string(pos));
    uint64_t len=be64(base+pos+8); std::string type(reinterpret_cast<const char*>(base+pos+16),128);
    const size_t nul=type.find('\0');
    if(nul!=std::string::npos) type.resize(nul);
    size_t data=pos+144;
    if(type=="ildg-binary-data") return {data,static_cast<size_t>(len)};
    pos=data+static_cast<size_t>((len+7ULL)&~7ULL);
  }
  throw std::runtime_error("LIME record ildg-binary-data not found");
}
struct MMapFile {
  int fd=-1; size_t size=0; unsigned char *ptr=nullptr;
  explicit MMapFile(const std::string &path) {
    fd=open(path.c_str(),O_RDONLY); if(fd<0) throw std::runtime_error("open failed: "+path);
    struct stat st{}; if(fstat(fd,&st)) throw std::runtime_error("fstat failed"); size=st.st_size;
    ptr=static_cast<unsigned char*>(mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0));
    if(ptr==MAP_FAILED) {ptr=nullptr; throw std::runtime_error("mmap failed");}
  }
  ~MMapFile(){if(ptr)munmap(ptr,size);if(fd>=0)close(fd);}
};
static std::vector<Mat3> read_ildg(const std::string &path,const Lattice &lat,LimePayload &payload) {
  MMapFile mm(path); payload=find_lime_payload(mm.ptr,mm.size);
  const size_t ndouble=lat.volume*4*18, expected=ndouble*8;
  if(payload.bytes!=expected) throw std::runtime_error("ILDG payload bytes="+std::to_string(payload.bytes)+", expected="+std::to_string(expected));
  std::vector<Mat3> u(lat.volume*4); const unsigned char *p=mm.ptr+payload.offset;
  #pragma omp parallel for schedule(static)
  for(long long k=0;k<static_cast<long long>(lat.volume*4*9);++k) {
    uint64_t ur=be64(p+16*k), ui=be64(p+16*k+8); double r,im;
    std::memcpy(&r,&ur,8); std::memcpy(&im,&ui,8);
    u[static_cast<size_t>(k)/9].v[k%9]=cf(static_cast<float>(r),static_cast<float>(im));
  }
  return u;
}

struct GaugeStats { double plaquette=0,max_unitarity=0,max_det=0; };
static GaugeStats gauge_stats(const std::vector<Mat3> &u,const Lattice &lat) {
  double psum=0.0,umax=0.0,dmax=0.0;
  #pragma omp parallel for reduction(+:psum) reduction(max:umax,dmax) schedule(static)
  for(long long si=0;si<static_cast<long long>(lat.volume);++si) {
    size_t s=si;
    for(int mu=0;mu<4;++mu) {
      Mat3 e=sub(mul(dagger(link(u,s,mu)),link(u,s,mu)),eye3()); umax=std::max(umax,std::sqrt(norm2(e)));
      dmax=std::max(dmax,std::abs(cd(determinant(link(u,s,mu)))-cd(1.0,0.0)));
      for(int nu=mu+1;nu<4;++nu) psum+=trace(loop4(u,lat,s,mu+1,nu+1,-mu-1,-nu-1)).real()/3.0;
    }
  }
  return {psum/(lat.volume*6.0),umax,dmax};
}

static void atomic_npy(const fs::path &path,const OpeData &d) {
  fs::create_directories(path.parent_path()); fs::path tmp=path; tmp += ".tmp."+std::to_string(getpid());
  std::ofstream f(tmp,std::ios::binary); if(!f) throw std::runtime_error("cannot write "+tmp.string());
  const std::string dict="{'descr': '<c16', 'fortran_order': False, 'shape': (2, 2, 4, 6, "+std::to_string(d.nz)+", "+std::to_string(d.nt)+"), }";
  size_t header_len=dict.size()+1, pre=10, pad=(16-(pre+header_len)%16)%16;
  std::string header=dict+std::string(pad,' ')+"\n"; uint16_t h=static_cast<uint16_t>(header.size());
  f.write("\x93NUMPY",6); char ver[2]={1,0}; f.write(ver,2); char hb[2]={char(h&255),char(h>>8)}; f.write(hb,2); f.write(header.data(),header.size());
  f.write(reinterpret_cast<const char*>(d.a.data()),d.a.size()*sizeof(cd)); f.close();
  if(!f) throw std::runtime_error("write failed "+tmp.string());
  fs::rename(tmp,path);
}
static std::string json_escape(const std::string &s) {
  std::string o; for(char c:s){if(c=='\\'||c=='\"')o+='\\';o+=c;} return o;
}
static void atomic_json(const fs::path &path,const std::string &input,const std::string &conf,
                        int nx,int ny,int nz,int nt,int zdir,int zcount,double tau,double eps,
                        const LimePayload &lp,const GaugeStats &before,const GaugeStats &after) {
  fs::path tmp=path; tmp += ".tmp."+std::to_string(getpid()); std::ofstream f(tmp);
  f<<std::setprecision(17)<<"{\n"
   <<"  \"schema\": \"gradient_flow_gluon_ope_v2\",\n"
   <<"  \"status\": \"flowed_bare_ope_observable\",\n"
   <<"  \"conf_id\": \""<<json_escape(conf)<<"\",\n"
   <<"  \"input_path\": \""<<json_escape(fs::absolute(input).string())<<"\",\n"
   <<"  \"input_scheme\": \"thin_link_no_hyp_no_smear\",\n"
   <<"  \"lime_payload_offset\": "<<lp.offset<<", \"lime_payload_bytes\": "<<lp.bytes<<",\n"
   <<"  \"lattice_xyzt\": ["<<nx<<","<<ny<<","<<nz<<","<<nt<<"],\n"
   <<"  \"flow\": {\"type\": \"Wilson\", \"integrator\": \"QUDA_RK3_coefficients_CPU_OpenMP\", \"tau_t_over_a2\": "<<tau<<", \"epsilon\": "<<eps<<", \"steps\": "<<std::llround(tau/eps)<<"},\n"
   <<"  \"operator\": {\"representation\": \"fundamental_closed_trace\", \"z_dir\": "<<zdir<<", \"z_values_a\": [";
  for(int z=0;z<zcount;++z){if(z)f<<",";f<<z;} f<<"], \"z_orientations\": [\"plus_z_raw\",\"minus_z_raw\",\"even_sum\",\"odd_difference\"], \"sum_kind\": \"full_spatial_sum_per_timeslice\", \"normalization\": \"bare_spatial_sum\", \"field_strength\": \"four_leaf_clover_-i_QminusQdagger_over_8\", \"field_projections\": [\"legacy_untraced\",\"traceless\"]},\n"
   <<"  \"channels\": {\"unpolarized_combined\": \"Mtiti-Mijij, Mtiti=M_ti_ti+M_tj_tj, Mijij=2*M_ij_ij\", \"helicity_components\": \"m=F_W_dualF; Mtiti=m_ti_dualti+m_tj_dualtj; Mijij=2*m_ij_dualij\", \"helicity_combined_euclidean\": \"Mtiti+Mijij (Euclidean time index t=3)\", \"helicity_paper_target\": \"odd_difference [Mtiti(z)+Mijij(z)]-[Mtiti(-z)+Mijij(-z)], arXiv:2207.08733v2 Eqs.(1),(2),(7)\"},\n"
   <<"  \"axes\": [\"field_projection\",\"channel\",\"z_orientation\",\"component\",\"z\",\"t\"],\n"
   <<"  \"axis_labels\": {\"field_projection\":[\"legacy_untraced\",\"traceless\"],\"channel\":[\"unpolarized\",\"helicity\"],\"z_orientation\":[\"plus_z_raw\",\"minus_z_raw\",\"even_sum\",\"odd_difference\"],\"component\":[\"combined\",\"Mtiti\",\"Mijij\",\"ti\",\"tj\",\"ij_single\"]},\n"
   <<"  \"dtype\": \"complex128\", \"shape\": [2,2,4,6,"<<zcount<<","<<nt<<"],\n"
   <<"  \"raw_gauge_stats\": {\"plaquette\": "<<before.plaquette<<", \"max_unitarity_frobenius\": "<<before.max_unitarity<<", \"max_abs_det_minus_1\": "<<before.max_det<<"},\n"
   <<"  \"flowed_gauge_stats\": {\"plaquette\": "<<after.plaquette<<", \"max_unitarity_frobenius\": "<<after.max_unitarity<<", \"max_abs_det_minus_1\": "<<after.max_det<<"},\n"
   <<"  \"theory_references\": {\"unpolarized\": \"arXiv:2510.26425v2 Eq. O(z)=M_tx;tx+M_ty;ty-2M_xy;xy\", \"helicity\": \"arXiv:2207.08733v2 Eqs.(1),(2),(7), raw components retained for Euclidean phase audit\"},\n"
   <<"  \"not_applied\": [\"nucleon_C2C3\",\"vacuum_subtraction\",\"renormalization\",\"flow_to_MS_matching\",\"LaMET_matching\"]\n} \n";
  f.close(); if(!f)throw std::runtime_error("metadata write failed"); fs::rename(tmp,path);
}

struct Args { std::string input,output,conf="unknown"; int nx=32,ny=32,nz=32,nt=96,zdir=2,zcount=25; double tau=0.5,eps=0.01; bool self=false; };
static Args parse(int argc,char **argv){Args a;for(int i=1;i<argc;++i){std::string x=argv[i];auto val=[&](){if(++i>=argc)throw std::runtime_error("missing value for "+x);return std::string(argv[i]);};
  if(x=="--input")a.input=val();else if(x=="--output")a.output=val();else if(x=="--conf-id")a.conf=val();else if(x=="--nx")a.nx=std::stoi(val());else if(x=="--ny")a.ny=std::stoi(val());else if(x=="--nz")a.nz=std::stoi(val());else if(x=="--nt")a.nt=std::stoi(val());else if(x=="--z-dir")a.zdir=std::stoi(val());else if(x=="--z-count")a.zcount=std::stoi(val());else if(x=="--tau")a.tau=std::stod(val());else if(x=="--epsilon")a.eps=std::stod(val());else if(x=="--self-test")a.self=true;else throw std::runtime_error("unknown argument "+x);}return a;}

static double max_diff(const Mat3 &a,const Mat3 &b){double m=0;for(int i=0;i<9;++i)m=std::max(m,std::abs(cd(a.v[i])-cd(b.v[i])));return m;}
static Mat3 exp_taylor(const Mat3 &a){Mat3 out=eye3(),term=eye3();for(int n=1;n<=24;++n){term=scale(mul(term,a),1.0f/n);out=add(out,term);}return out;}
static Mat3 test_generator(int key,float size){
  Mat3 a{}; float x=size*(1.0f+0.07f*(key%7));
  a(0,0)=cf(0,0.4f*x);a(1,1)=cf(0,-0.1f*x);a(2,2)=cf(0,-0.3f*x);
  a(0,1)=cf(0.30f*x,0.11f*x);a(1,0)=-std::conj(a(0,1));
  a(0,2)=cf(-0.17f*x,0.09f*x);a(2,0)=-std::conj(a(0,2));
  a(1,2)=cf(0.13f*x,-0.21f*x);a(2,1)=-std::conj(a(1,2));return a;
}
static void self_test(){
  Mat3 a=test_generator(3,0.2f);double exp_error=max_diff(exp_antiherm(a),exp_taylor(a));
  if(exp_error>3e-6)throw std::runtime_error("SU3 exponential self-test failed: "+std::to_string(exp_error));
  Lattice lat(2,2,2,2); std::vector<Mat3> identity(lat.volume*4,eye3());
  wilson_flow(identity,lat,2,0.01f); auto fi=make_fields(identity,lat); OpeData oi=compute_ope(identity,lat,fi,2,2);
  double m=0;for(cd x:oi.a)m=std::max(m,std::abs(x)); GaugeStats si=gauge_stats(identity,lat);
  if(m>1e-12||si.max_unitarity>2e-6||std::abs(si.plaquette-1.0)>2e-6)throw std::runtime_error("identity self-test failed");

  std::vector<Mat3> u(lat.volume*4),g(lat.volume),ug(lat.volume*4);
  for(size_t s=0;s<lat.volume;++s){g[s]=exp_antiherm(test_generator(static_cast<int>(s),0.16f));for(int mu=0;mu<4;++mu)u[4*s+mu]=exp_antiherm(test_generator(static_cast<int>(4*s+mu),0.22f));}
  for(size_t s=0;s<lat.volume;++s)for(int mu=0;mu<4;++mu)ug[4*s+mu]=mul(g[s],mul(u[4*s+mu],dagger(g[lat.plus[mu][s]])));
  auto f0=make_fields(u,lat),fg=make_fields(ug,lat);OpeData o0=compute_ope(u,lat,f0,2,2),og=compute_ope(ug,lat,fg,2,2);
  double cov=0,scale0=0;for(size_t k=0;k<o0.a.size();++k){cov=std::max(cov,std::abs(o0.a[k]-og.a[k]));scale0=std::max(scale0,std::abs(o0.a[k]));}
  if(cov>2e-4*std::max(1.0,scale0))throw std::runtime_error("OPE gauge-covariance self-test failed");
  double algebra=0;
  for(int p=0;p<2;++p)for(int c=0;c<2;++c)for(int z=0;z<2;++z)for(int t=0;t<2;++t){
    for(int o=0;o<4;++o){
      algebra=std::max(algebra,std::abs(o0.at(p,c,o,1,z,t)-o0.at(p,c,o,3,z,t)-o0.at(p,c,o,4,z,t)));
      algebra=std::max(algebra,std::abs(o0.at(p,c,o,2,z,t)-2.0*o0.at(p,c,o,5,z,t)));
      const cd wanted=c?(o0.at(p,c,o,1,z,t)+o0.at(p,c,o,2,z,t)):(o0.at(p,c,o,1,z,t)-o0.at(p,c,o,2,z,t));
      algebra=std::max(algebra,std::abs(o0.at(p,c,o,0,z,t)-wanted));
    }
    for(int k=0;k<6;++k){
      const cd plus=o0.at(p,c,0,k,z,t),minus=o0.at(p,c,1,k,z,t);
      algebra=std::max(algebra,std::abs(o0.at(p,c,2,k,z,t)-(plus+minus)));
      algebra=std::max(algebra,std::abs(o0.at(p,c,3,k,z,t)-(plus-minus)));
      if(z==0)algebra=std::max(algebra,std::abs(plus-minus));
    }
  }
  if(algebra>1e-11*std::max(1.0,scale0))throw std::runtime_error("OPE component/orientation algebra self-test failed");
  GaugeStats before=gauge_stats(u,lat);wilson_flow(u,lat,1,0.01f);wilson_flow(ug,lat,1,0.01f);GaugeStats after=gauge_stats(u,lat);
  double flow_cov=0;for(size_t s=0;s<lat.volume;++s)for(int mu=0;mu<4;++mu){Mat3 want=mul(g[s],mul(u[4*s+mu],dagger(g[lat.plus[mu][s]])));flow_cov=std::max(flow_cov,max_diff(want,ug[4*s+mu]));}
  if(flow_cov>2e-5||after.plaquette+1e-7<before.plaquette)throw std::runtime_error("flow covariance/monotonicity self-test failed");
  std::cout<<"SELF_TEST_OK max_identity_ope="<<m<<" exp_error="<<exp_error<<" ope_covariance="<<cov<<" component_algebra="<<algebra<<" flow_covariance="<<flow_cov<<" plaquette_before="<<before.plaquette<<" plaquette_after="<<after.plaquette<<"\n";
}

int main(int argc,char **argv){try{Args a=parse(argc,argv);if(a.self){self_test();return 0;}if(a.input.empty()||a.output.empty())throw std::runtime_error("--input and --output are required");
  if(a.zdir<0||a.zdir>2||a.zcount<1||a.tau<0||a.eps<=0)throw std::runtime_error("invalid physics arguments");
  const long long steps=std::llround(a.tau/a.eps);if(std::abs(a.tau-steps*a.eps)>5e-12)throw std::runtime_error("tau/epsilon must be an integer");
  Lattice lat(a.nx,a.ny,a.nz,a.nt);LimePayload lp;std::cout<<"reading "<<a.input<<std::endl;auto u=read_ildg(a.input,lat,lp);GaugeStats before=gauge_stats(u,lat);
  std::cout<<"raw_plaquette="<<before.plaquette<<" raw_unitarity="<<before.max_unitarity<<std::endl;
  wilson_flow(u,lat,static_cast<int>(steps),static_cast<float>(a.eps));GaugeStats after=gauge_stats(u,lat);
  std::cout<<"flowed_plaquette="<<after.plaquette<<" flowed_unitarity="<<after.max_unitarity<<std::endl;
  auto fields=make_fields(u,lat);auto ope=compute_ope(u,lat,fields,a.zdir,a.zcount);
  fs::path base(a.output);atomic_npy(base.string()+".npy",ope);atomic_json(base.string()+".json",a.input,a.conf,a.nx,a.ny,a.nz,a.nt,a.zdir,a.zcount,a.tau,a.eps,lp,before,after);
  std::cout<<"OUTPUT_OK base="<<a.output<<std::endl;return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 2;}}
