import os
import shutil
import numpy as np
import pytest

import yt
from yt.frontends.pasta.api import PastaDataset
from yt.geometry.api import Geometry
from yt.testing import assert_allclose_units, requires_file

PASTA_1D_AMR = "/home/jwise/codes/Pasta/output_amr/bin/data.prim.00001.binary"
PASTA_3D_AMR = "/home/jwise/codes/Pasta/regression/build/bin/data.prim.00000.binary"


def _write_pasta_binary(
    filename,
    coord_type="Cartesian",
    nblocks=1,
    ncells=(8, 8, 8),
    vars_to_write=("fluid_den_prim", "fluid_vel_1", "fluid_vel_2", "fluid_vel_3", "fluid_pressure"),
    num_face_field=0,
    num_edge_field=0,
    cycle=0,
    time=0.0,
    x1_range=(0.0, 1.0),
    x2_range=(0.0, 1.0),
    x3_range=(0.0, 1.0),
):
    nc1, nc2, nc3 = ncells
    nvar = len(vars_to_write)
    var_str = "  ".join(vars_to_write)

    header = (
        f"# Pasta output at global time = {time:.16e}\n"
        f"# coord_type = {coord_type}\n"
        f"# cycle = {cycle}\n"
        "# grid_type_1 = 1\n"
        "# grid_type_2 = 1\n"
        "# grid_type_3 = 1\n"
        "# density unit/cgs = 1.0000000000000000e+00\n"
        "# length unit/cgs = 1.0000000000000000e+00\n"
        "# time unit/cgs = 1.0000000000000000e+00\n"
        "# velocity unit/cgs = 1.0000000000000000e+00\n"
        "# pressure unit/cgs = 1.0000000000000000e+00\n"
        "# temperature unit/cgs = 1.0000000000000000e+00\n"
        "# radiation energy density unit/cgs = 1.0000000000000000e+00\n"
        "# radiation flux unit/cgs = 1.0000000000000000e+00\n"
        "# size of coordinate data = 8\n"
        "# size of variable data = 8\n"
        f"# number of variables = {nvar}\n"
        "# number of MPI ranks = 1\n"
        f"# variables:  {var_str}\n"
        "# number of frequencies = 0\n"
    )
    if num_face_field > 0:
        header += f"# face_field = {num_face_field}\n"
    if num_edge_field > 0:
        header += f"# edge_field = {num_edge_field}\n"

    with open(filename, "wb") as f:
        f.write(np.int32(1).tobytes())
        f.write(header.encode("ascii"))
        binfo = np.array([0, nblocks, nvar, nc3, nc2, nc1], dtype="<i4")
        f.write(binfo.tobytes())

        tot_cells = nc1 * nc2 * nc3
        for b in range(nblocks):
            # Coordinates
            x3f = np.linspace(x3_range[0], x3_range[1], nc3 + 1, dtype="<f8")
            x3c = 0.5 * (x3f[:-1] + x3f[1:])
            x2f = np.linspace(x2_range[0], x2_range[1], nc2 + 1, dtype="<f8")
            x2c = 0.5 * (x2f[:-1] + x2f[1:])
            dx1 = (x1_range[1] - x1_range[0]) / nblocks
            x1f = np.linspace(x1_range[0] + b * dx1, x1_range[0] + (b + 1) * dx1, nc1 + 1, dtype="<f8")
            x1c = 0.5 * (x1f[:-1] + x1f[1:])

            f.write(x3f.tobytes())
            f.write(x3c.tobytes())
            f.write(x2f.tobytes())
            f.write(x2c.tobytes())
            f.write(x1f.tobytes())
            f.write(x1c.tobytes())

            # Fluid variables
            for v_idx, v_name in enumerate(vars_to_write):
                val = (b + 1.0) * (v_idx + 1.0)
                arr = np.full(tot_cells, val, dtype="<f8")
                f.write(arr.tobytes())

            # Staggered face fields if any
            if num_face_field > 0:
                f1 = np.ones((nc3, nc2, nc1 + 1), dtype="<f8") * 0.1
                f2 = np.ones((nc3, nc2 + 1, nc1), dtype="<f8") * 0.2
                f3 = np.ones((nc3 + 1, nc2, nc1), dtype="<f8") * 0.3
                f.write(f1.tobytes())
                f.write(f2.tobytes())
                f.write(f3.tobytes())


def _write_pasta_particle(
    filename,
    coord_type="Cartesian",
    nblocks=1,
    npar_per_block=(4,),
    cycle=0,
    time=0.0,
):
    total_npar = sum(npar_per_block)
    header = (
        f"# Pasta output at global time = {time:.16e}\n"
        f"# coord_type = {coord_type}\n"
        f"# cycle = {cycle}\n"
        f"# particle count = {total_npar}\n"
        "# int properties:  pid blk\n"
        "# real properties: x1 x2 x3 v1 v2 v3 mass\n"
        f"# number of blocks = {nblocks}\n"
    )

    with open(filename, "wb") as f:
        f.write(np.int32(1).tobytes())
        f.write(header.encode("ascii"))
        f.write(np.array([2, 7, 8, 8], dtype="<i4").tobytes())
        f.write(np.int64(total_npar).tobytes())
        f.write(np.int64(nblocks).tobytes())
        f.write(np.array(npar_per_block, dtype="<i8").tobytes())

        # int properties: pid, blk
        pid = np.arange(total_npar, dtype="<i8")
        blk_list = []
        for b, count in enumerate(npar_per_block):
            blk_list.extend([b] * count)
        blk = np.array(blk_list, dtype="<i8")
        f.write(pid.tobytes())
        f.write(blk.tobytes())

        # real properties: x1, x2, x3, v1, v2, v3, mass
        x1 = np.linspace(0.1, 0.9, total_npar, dtype="<f8")
        x2 = np.linspace(0.2, 0.8, total_npar, dtype="<f8")
        x3 = np.linspace(0.3, 0.7, total_npar, dtype="<f8")
        v1 = np.ones(total_npar, dtype="<f8") * 10.0
        v2 = np.ones(total_npar, dtype="<f8") * 20.0
        v3 = np.ones(total_npar, dtype="<f8") * 30.0
        mass = np.ones(total_npar, dtype="<f8") * 0.05
        for arr in [x1, x2, x3, v1, v2, v3, mass]:
            f.write(arr.tobytes())


@requires_file(PASTA_1D_AMR)
def test_pasta_1d_amr():
    ds = yt.load(PASTA_1D_AMR)
    assert isinstance(ds, PastaDataset)
    assert ds.dimensionality == 1
    assert ds.geometry == "cartesian"
    assert ds.index.num_grids == 16
    assert ds.index.max_level == 2

    ad = ds.all_data()
    dens = ad["gas", "density"]
    assert dens.size == 512
    assert ("gas", "pressure") in ds.derived_field_list
    assert ("gas", "velocity_x") in ds.derived_field_list
    assert ("gas", "temperature") in ds.derived_field_list
    assert ("gas", "sound_speed") in ds.derived_field_list


@requires_file(PASTA_3D_AMR)
def test_pasta_3d_amr():
    ds = yt.load(PASTA_3D_AMR)
    assert isinstance(ds, PastaDataset)
    assert ds.dimensionality == 3
    assert ds.geometry == "cartesian"
    assert ds.index.num_grids == 18
    assert ds.index.max_level == 1

    ad = ds.all_data()
    dens = ad["gas", "density"]
    assert dens.size == 147456


def test_pasta_cylindrical(tmp_path):
    fn = str(tmp_path / "data.prim.00000.binary")
    _write_pasta_binary(
        fn,
        coord_type="Cylindrical",
        nblocks=2,
        ncells=(4, 4, 4),
        x1_range=(0.1, 1.0),
        x2_range=(0.0, 2.0 * np.pi),
        x3_range=(-1.0, 1.0),
    )
    ds = yt.load(fn)
    assert isinstance(ds, PastaDataset)
    assert ds.geometry == "cylindrical"
    assert ds.coordinates.axis_order == ("r", "theta", "z")

    ad = ds.all_data()
    assert ("gas", "velocity_r") in ds.derived_field_list
    assert ("gas", "velocity_theta") in ds.derived_field_list
    assert ("gas", "velocity_z") in ds.derived_field_list
    assert ad["index", "r"].size == 2 * 64
    assert ad["index", "cell_volume"].size == 2 * 64


def test_pasta_spherical(tmp_path):
    fn = str(tmp_path / "data.prim.00000.binary")
    _write_pasta_binary(
        fn,
        coord_type="Spherical",
        nblocks=1,
        ncells=(4, 4, 4),
        x1_range=(0.1, 1.0),
        x2_range=(0.0, np.pi),
        x3_range=(0.0, 2.0 * np.pi),
    )
    ds = yt.load(fn)
    assert isinstance(ds, PastaDataset)
    assert ds.geometry == "spherical"
    assert ds.coordinates.axis_order == ("r", "theta", "phi")

    ad = ds.all_data()
    assert ("gas", "velocity_r") in ds.derived_field_list
    assert ("gas", "velocity_theta") in ds.derived_field_list
    assert ("gas", "velocity_phi") in ds.derived_field_list
    assert ad["index", "r"].size == 64
    assert ad["index", "cell_volume"].size == 64


def test_pasta_magnetic_fields(tmp_path):
    fn = str(tmp_path / "data.prim.00000.binary")
    _write_pasta_binary(
        fn,
        coord_type="Cartesian",
        nblocks=1,
        ncells=(4, 4, 4),
        vars_to_write=("fluid_den_prim", "field_bcc_1", "field_bcc_2", "field_bcc_3"),
        num_face_field=1,
    )
    ds = yt.load(fn)
    assert ("gas", "magnetic_field_x") in ds.derived_field_list
    assert ("gas", "magnetic_field_y") in ds.derived_field_list
    assert ("gas", "magnetic_field_z") in ds.derived_field_list

    ad = ds.all_data()
    bx = ad["gas", "magnetic_field_x"]
    assert str(bx.units) == "G"
    assert bx.size == 64

    # Read staggered face field
    bf1 = ad["pasta", "field_bf_1"]
    assert bf1.size == 64


def test_pasta_particles(tmp_path):
    bin_dir = tmp_path / "bin"
    par_dir = tmp_path / "par"
    bin_dir.mkdir()
    par_dir.mkdir()

    grid_file = str(bin_dir / "data.prim.00001.binary")
    par_file = str(par_dir / "data.00001.par")

    _write_pasta_binary(
        grid_file,
        coord_type="Cartesian",
        nblocks=2,
        ncells=(4, 4, 4),
        cycle=1,
        time=0.1,
    )
    _write_pasta_particle(
        par_file,
        coord_type="Cartesian",
        nblocks=2,
        npar_per_block=(3, 5),
        cycle=1,
        time=0.1,
    )

    # 1. Load via grid binary file
    ds = yt.load(grid_file)
    assert ds._has_particles is True
    assert ds.index.num_grids == 2
    assert ds.index.grids[0].NumberOfParticles == 3
    assert ds.index.grids[1].NumberOfParticles == 5

    ad = ds.all_data()
    px = ad["io", "particle_position_x"]
    py = ad["io", "particle_position_y"]
    pz = ad["io", "particle_position_z"]
    mass = ad["io", "particle_mass"]
    pid = ad["io", "particle_index"]

    assert px.size == 8
    assert py.size == 8
    assert pz.size == 8
    assert mass.size == 8
    assert pid.size == 8
    assert ad["io", "particle_position"].shape == (8, 3)
    assert ad["io", "particle_velocity"].shape == (8, 3)

    # Spatial region selection
    reg = ds.region([0.25, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 1.0, 1.0])
    # block 0 is in x [0, 0.5], block 1 is in [0.5, 1.0]
    reg_px = reg["io", "particle_position_x"]
    assert reg_px.size == 3

    # 2. Load via companion par file directly
    ds_par = yt.load(par_file)
    assert ds_par._has_particles is True
    assert ds_par.index.num_grids == 2
    ad_par = ds_par.all_data()
    assert ad_par["io", "particle_position_x"].size == 8

    # 3. Standalone par file (without companion grid)
    standalone_par = str(tmp_path / "standalone.par")
    shutil.copy(par_file, standalone_par)
    ds_standalone = yt.load(standalone_par)
    assert ds_standalone._has_particles is True
    assert ds_standalone.index.num_grids == 1
    ad_standalone = ds_standalone.all_data()
    assert ad_standalone["io", "particle_position_x"].size == 8
