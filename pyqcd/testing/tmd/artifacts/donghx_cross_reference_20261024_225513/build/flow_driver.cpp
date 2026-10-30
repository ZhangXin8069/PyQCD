#define main donghx_archived_cli_main
#include "/root/PyQCD/refer/donghx/_Gradient_Flow_GluonPDF/flowed_gluon_ope.cpp"
#undef main

static void write_flowed_links(const fs::path &path,
                               const std::vector<Mat3> &links) {
  fs::path tmp = path;
  tmp += ".tmp";
  std::ofstream stream(tmp, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open flow output");
  for (const Mat3 &matrix : links) {
    for (int component = 0; component < 9; ++component) {
      const float real = matrix.v[component].real();
      const float imag = matrix.v[component].imag();
      stream.write(reinterpret_cast<const char *>(&real), sizeof(float));
      stream.write(reinterpret_cast<const char *>(&imag), sizeof(float));
    }
  }
  stream.close();
  if (!stream) throw std::runtime_error("cannot write flow output");
  fs::rename(tmp, path);
}

int main(int argc, char **argv) {
  try {
    std::string input, output;
    int nx = 4, ny = 4, nz = 4, nt = 8;
    double tau = 0.0, epsilon = 0.01;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      auto value = [&]() {
        if (++i >= argc) throw std::runtime_error("missing value for " + key);
        return std::string(argv[i]);
      };
      if (key == "--input") input = value();
      else if (key == "--output") output = value();
      else if (key == "--nx") nx = std::stoi(value());
      else if (key == "--ny") ny = std::stoi(value());
      else if (key == "--nz") nz = std::stoi(value());
      else if (key == "--nt") nt = std::stoi(value());
      else if (key == "--tau") tau = std::stod(value());
      else if (key == "--epsilon") epsilon = std::stod(value());
      else throw std::runtime_error("unknown argument " + key);
    }
    if (input.empty() || output.empty() || tau < 0.0 || epsilon <= 0.0)
      throw std::runtime_error("invalid arguments");
    const long long steps = std::llround(tau / epsilon);
    if (std::abs(tau - steps * epsilon) > 5e-12)
      throw std::runtime_error("tau/epsilon must be an integer");
    Lattice lattice(nx, ny, nz, nt);
    LimePayload payload;
    auto links = read_ildg(input, lattice, payload);
    wilson_flow(links, lattice, static_cast<int>(steps), static_cast<float>(epsilon));
    write_flowed_links(output, links);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 2;
  }
}
