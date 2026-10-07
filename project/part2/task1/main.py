import consistent_mass_matrix
import lumped_mass_matrix
import unstable_galerkin


def main():
    h = 0.1
    CFL_list = (0.01, 0.05, 0.1, 0.2, 0.5, 0.8)

    print("Running consistent mass matrix simulation...")
    consistent_mass_matrix.main(h, CFL_list)
    print("Running lumped mass matrix simulation...")
    lumped_mass_matrix.main(h, CFL_list)
    print("Running unstable Galerkin simulation...")
    unstable_galerkin.main(h, CFL_list)


if __name__ == "__main__":
    main()
