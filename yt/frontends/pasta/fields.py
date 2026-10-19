import numpy as np

from yt._typing import KnownFieldsT
from yt.fields.field_info_container import FieldInfoContainer
from yt.utilities.physical_constants import kboltz, mh

b_units = "code_magnetic"
pres_units = "code_mass / (code_length * code_time**2)"
rho_units = "code_mass / code_length**3"
vel_units = "code_length / code_time"
mom_units = "code_mass / (code_length**2 * code_time)"
pot_units = "code_length**2 / code_time**2"


def velocity_field(j):
    def _velocity(data):
        return data["pasta", f"fluid_mom_{j}"] / data["gas", "density"]

    return _velocity


class PastaFieldInfo(FieldInfoContainer):
    known_other_fields: KnownFieldsT = (
        ("fluid_den_prim", (rho_units, ["density"], None)),
        ("fluid_den_cons", (rho_units, ["density"], None)),
        ("fluid_vel_1", (vel_units, [], None)),
        ("fluid_vel_2", (vel_units, [], None)),
        ("fluid_vel_3", (vel_units, [], None)),
        ("fluid_mom_1", (mom_units, [], None)),
        ("fluid_mom_2", (mom_units, [], None)),
        ("fluid_mom_3", (mom_units, [], None)),
        ("fluid_pressure", (pres_units, ["pressure"], None)),
        ("fluid_internale", (pres_units, ["thermal_energy_density"], None)),
        ("fluid_energy", (pres_units, ["total_energy_per_volume"], None)),
        ("temperature", ("K", [], None)),
        ("field_bcc_1", (b_units, [], None)),
        ("field_bcc_2", (b_units, [], None)),
        ("field_bcc_3", (b_units, [], None)),
        ("field_bf_1", (b_units, [], None)),
        ("field_bf_2", (b_units, [], None)),
        ("field_bf_3", (b_units, [], None)),
        ("gravity_phi", (pot_units, ["gravitational_potential"], None)),
        ("gravity_phi_old", (pot_units, [], None)),
        ("particle_rho", (rho_units, [], None)),
    )

    known_particle_fields: KnownFieldsT = (
        ("particle_position_x", ("code_length", [], None)),
        ("particle_position_y", ("code_length", [], None)),
        ("particle_position_z", ("code_length", [], None)),
        ("particle_velocity_x", (vel_units, [], None)),
        ("particle_velocity_y", (vel_units, [], None)),
        ("particle_velocity_z", (vel_units, [], None)),
        ("particle_mass", ("code_mass", [], None)),
        ("particle_index", ("", [], None)),
        ("x1", ("code_length", [], None)),
        ("x2", ("code_length", [], None)),
        ("x3", ("code_length", [], None)),
        ("v1", (vel_units, [], None)),
        ("v2", (vel_units, [], None)),
        ("v3", (vel_units, [], None)),
        ("mass", ("code_mass", [], None)),
        ("pid", ("", [], None)),
        ("blk", ("", [], None)),
    )

    def setup_fluid_fields(self):
        from yt.fields.magnetic_field import setup_magnetic_field_aliases

        unit_system = self.ds.unit_system

        # Velocity and momentum aliasing based on active axis ordering
        for i, comp in enumerate(self.ds.coordinates.axis_order):
            vel_field = ("pasta", f"fluid_vel_{i + 1}")
            mom_field = ("pasta", f"fluid_mom_{i + 1}")
            if vel_field in self.field_list:
                self.add_output_field(
                    vel_field, sampling_type="cell", units=vel_units
                )
                self.alias(
                    ("gas", f"velocity_{comp}"),
                    vel_field,
                    units=unit_system["velocity"],
                )
            elif mom_field in self.field_list:
                self.add_output_field(
                    mom_field, sampling_type="cell", units=mom_units
                )
                self.add_field(
                    ("gas", f"velocity_{comp}"),
                    sampling_type="cell",
                    function=velocity_field(i + 1),
                    units=unit_system["velocity"],
                )

            if mom_field in self.field_list:
                self.add_output_field(
                    mom_field, sampling_type="cell", units=mom_units
                )
                self.alias(
                    ("gas", f"momentum_{comp}"),
                    mom_field,
                    units=unit_system["momentum_density"],
                )

        # Thermal energy
        if ("pasta", "fluid_internale") in self.field_list:
            def _specific_thermal_energy(data):
                return (
                    data["pasta", "fluid_internale"] / data["gas", "density"]
                )

            self.add_field(
                ("gas", "specific_thermal_energy"),
                sampling_type="cell",
                function=_specific_thermal_energy,
                units=unit_system["specific_energy"],
            )
        elif ("pasta", "fluid_pressure") in self.field_list:
            def _specific_thermal_energy(data):
                return (
                    data["pasta", "fluid_pressure"]
                    / (data.ds.gamma - 1.0)
                    / data["gas", "density"]
                )

            self.add_field(
                ("gas", "specific_thermal_energy"),
                sampling_type="cell",
                function=_specific_thermal_energy,
                units=unit_system["specific_energy"],
            )

        # Sound speed
        if ("pasta", "fluid_pressure") in self.field_list:
            def _sound_speed(data):
                return np.sqrt(
                    data.ds.gamma
                    * data["gas", "pressure"]
                    / data["gas", "density"]
                )

            self.add_field(
                ("gas", "sound_speed"),
                sampling_type="cell",
                function=_sound_speed,
                units=unit_system["velocity"],
            )

        # Temperature
        if ("pasta", "temperature") in self.field_list:
            self.alias(
                ("gas", "temperature"),
                ("pasta", "temperature"),
                units=unit_system["temperature"],
            )
        elif ("pasta", "fluid_pressure") in self.field_list:
            def _temperature(data):
                return (
                    (data["gas", "pressure"] / data["gas", "density"])
                    * data.ds.mu
                    * mh
                    / kboltz
                )

            self.add_field(
                ("gas", "temperature"),
                sampling_type="cell",
                function=_temperature,
                units=unit_system["temperature"],
            )

        # Magnetic fields
        b_fields = [f"field_bcc_{i + 1}" for i in range(3)]
        if any(("pasta", bf) in self.field_list for bf in b_fields):
            setup_magnetic_field_aliases(self, "pasta", b_fields)

    def setup_particle_fields(self, ptype):
        super().setup_particle_fields(ptype)
        unit_system = self.ds.unit_system

        if (ptype, "mass") in self.field_list and (ptype, "particle_mass") not in self:
            self.alias((ptype, "particle_mass"), (ptype, "mass"), units=unit_system["mass"])
        if (ptype, "pid") in self.field_list and (ptype, "particle_index") not in self:
            self.alias((ptype, "particle_index"), (ptype, "pid"), units="")

        if self.ds.geometry.name == "cylindrical":
            for ax, raw in [("r", "x1"), ("theta", "x2"), ("z", "x3")]:
                if (ptype, raw) in self.field_list and (ptype, f"particle_position_{ax}") not in self:
                    self.alias((ptype, f"particle_position_{ax}"), (ptype, raw), units=unit_system["length"])
            for ax, raw in [("r", "v1"), ("theta", "v2"), ("z", "v3")]:
                if (ptype, raw) in self.field_list and (ptype, f"particle_velocity_{ax}") not in self:
                    self.alias((ptype, f"particle_velocity_{ax}"), (ptype, raw), units=unit_system["velocity"])
        elif self.ds.geometry.name == "spherical":
            for ax, raw in [("r", "x1"), ("theta", "x2"), ("phi", "x3")]:
                if (ptype, raw) in self.field_list and (ptype, f"particle_position_{ax}") not in self:
                    self.alias((ptype, f"particle_position_{ax}"), (ptype, raw), units=unit_system["length"])
            for ax, raw in [("r", "v1"), ("theta", "v2"), ("phi", "v3")]:
                if (ptype, raw) in self.field_list and (ptype, f"particle_velocity_{ax}") not in self:
                    self.alias((ptype, f"particle_velocity_{ax}"), (ptype, raw), units=unit_system["velocity"])

