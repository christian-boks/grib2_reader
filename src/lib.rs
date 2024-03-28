//! Read a GRIB2 file and search for data based on parameter and level values. The results can either be decoded or extracted as a binary blob so it can be saved to a separate file.
//! Currently only some of the functionality is implemented.

use bitstream_io::{BigEndian, BitRead, BitReader};
use error::Grib2Error;

#[cfg(feature = "json")]
use serde::Deserialize;
#[cfg(feature = "json")]
use serde::Serialize;

use std::io::Cursor;
use std::io::SeekFrom;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, BufReader};
pub mod error;

/// The star of the show
pub struct Grib2Reader {
    pub reader: BufReader<File>,
    offset: u64,
}

#[derive(Debug, Default)]
/// Grib file representation
pub struct Grib {
    pub length: u64,
    pub discipline: u8,
    pub identification: Option<Identification>,
    pub grid_definition: Option<GridDefinition>,
    pub product_definition: Vec<ProductDefinition>,
    pub data_representation: Vec<DataRepresentation>,
    pub bitmap: Vec<Bitmap>,
    pub data: Vec<Data>,
}

#[derive(Debug)]
enum GribResult {
    Length(u64),
    Grib(Grib),
}

#[derive(Debug, Clone)]
/// Identification section
pub struct Identification {
    pub identification_of_originating_generating_centre: u16,
    pub identification_of_originating_generating_sub_centre: u16,
    pub grib_master_tables_version_number: u8,
    pub grib_local_tables_version_number: u8,
    pub significance_of_reference_time: u8,
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub production_status_of_processed_data: u8,
    pub type_of_processed_data: u8,
}

#[derive(Debug, Clone)]
/// Grid Definition
pub struct GridDefinition {
    pub source_of_grid_definition: u8,
    pub number_of_data_points: u32,
    pub number_of_octets_for_optional_list_of_numbers_defining_number_of_points: u8,
    pub interpretation_of_list_of_numbers_defining_number_of_points: u8,
    pub grid_definition_template_number: u16,
    pub template: GridDefinitionTemplate,
}

#[derive(Debug, Clone, Default)]
pub struct LambertConformalTemplate {
    pub shape_of_the_earth: u8,
    pub scale_factor_of_radius_of_spherical_earth: u8,
    pub scale_value_of_radius_of_spherical_earth: u32,
    pub scale_factor_of_major_axis_of_oblate_spheroid_earth: u8,
    pub scaled_value_of_major_axis_of_oblate_spheroid_earth: u32,
    pub scale_factor_of_minor_axis_of_oblate_spheroid_earth: u8,
    pub scaled_value_of_minor_axis_of_oblate_spheroid_earth: u32,
    pub nx_number_of_points_along_the_x_axis: u32,
    pub ny_number_of_points_along_the_y_axis: u32,
    pub la1_latitude_of_first_grid_point: i32,
    pub lo1_longitude_of_first_grid_point: i32,
    pub resolution_and_component_flags: u8,
    pub lad_latitude_where_dx_and_dy_are_specified: i32,
    pub lov_longitude_of_meridian_parallel_to_y_axis_along_which_latitude_increases_as_the_y_coordinate_increases: i32,
    pub dx_x_direction_grid_length: u32,
    pub dy_y_direction_grid_length: u32,
    pub projection_centre_flag: u8,
    pub scanning_mode: u8,
    pub latin_1_first_latitude_from_the_pole_at_which_the_secant_cone_cuts_the_sphere: i32,
    pub latin_2_second_latitude_from_the_pole_at_which_the_secant_cone_cuts_the_sphere: i32,
    pub latitude_of_the_southern_pole_of_projection: i32,
    pub longitude_of_the_southern_pole_of_projection: i32,
}

#[derive(Debug, Clone)]
pub enum GridDefinitionTemplate {
    LambertConformal(LambertConformalTemplate),
    Unknown,
}

#[derive(Debug, Clone)]
/// Product Definition
pub struct ProductDefinition {
    pub number_of_coordinate_values_after_template: u16,
    pub product_definition_template_number: u16,
    pub template: ProductDefinitionTemplate,
}

#[derive(Debug, Clone)]
pub enum ProductDefinitionTemplate {
    Id1(Id1ProductDefinitionTemplate),
    Id11(Id11ProductDefinitionTemplate),
    Unknown,
}

#[derive(Debug, Clone, Default)]
pub struct Id1ProductDefinitionTemplate {
    pub parameter_category: u8,
    pub parameter_number: u8,
    pub type_of_generating_process: u8,
    pub background_generating_process_identifier_defined_by_originating_centre: u8,
    pub forecast_generating_process_identified: u8,
    pub hours_after_reference_time_data_cutoff: u16,
    pub minutes_after_reference_time_data_cutoff: u8,
    pub indicator_of_unit_of_time_range: u8,
    pub forecast_time_in_units_defined_by_octet_18: u32,
    pub type_of_first_fixed_surface: u8,
    pub scale_factor_of_first_fixed_surface: u8,
    pub scaled_value_of_first_fixed_surface: u32,
    pub type_of_second_fixed_surfaced: u8,
    pub scale_factor_of_second_fixed_surface: u8,
    pub scaled_value_of_second_fixed_surfaces: u32,
    pub type_of_ensemble_forecast: u8,
    pub perturbation_number: u8,
    pub number_of_forecasts_in_ensemble: u8,
}

#[derive(Debug, Clone, Default)]
pub struct Id11ProductDefinitionTemplate {
    pub parameter_category: u8,
    pub parameter_number: u8,
    pub type_of_generating_process: u8,
    pub background_generating_process_identifier_defined_by_originating_centre: u8,
    pub forecast_generating_process_identifier: u8,
    pub hours_after_reference_time_data_cutoff: u16,
    pub minutes_after_reference_time_data_cutoff: u8,
    pub indicator_of_unit_of_time_range: u8,
    pub forecast_time_in_units_defined_by_octet_18: u32,
    pub type_of_first_fixed_surface: u8,
    pub scale_factor_of_first_fixed_surface: u8,
    pub scaled_value_of_first_fixed_surface: u32,
    pub type_of_second_fixed_surfaced: u8,
    pub scale_factor_of_second_fixed_surface: u8,
    pub scaled_value_of_second_fixed_surfaces: u32,
    pub type_of_ensemble_forecast: u8,
    pub perturbation_number: u8,
    pub number_of_forecasts_in_ensemble: u8,
    pub year_of_end_of_overall_time_interval: u16,
    pub month_of_end_of_overall_time_interval: u8,
    pub day_of_end_of_overall_time_interval: u8,
    pub hour_of_end_of_overall_time_interval: u8,
    pub minute_of_end_overall_time_interval: u8,
    pub second_of_end_of_overall_time_interval: u8,
    pub n_number_of_time_ranges_specifications_describing_the_time_intervals_used_to_calculate_the_statistically_processed_field: u8,
    pub total_number_of_data_values_missing_in_the_statistical_process: u32,
}

#[derive(Debug, Clone)]
/// Product Definition
pub struct DataRepresentation {
    pub number_of_data_points: u32,
    pub data_representation_template_number: u16,
    pub template: DataRepresentationTemplate,
}

#[derive(Debug, Clone)]
pub enum DataRepresentationTemplate {
    SimplePacking(SimplePackingTemplate),
    Unknown,
}

#[derive(Debug, Clone)]
pub struct SimplePackingTemplate {
    reference_value: f32,
    binary_scale_factor: i16,
    decimal_scale_factor: i16,
    number_of_bits_used_for_each_packed_value: u8,
    type_of_original_field_values: u8,
}

#[derive(Debug)]
/// Bit-map section
pub struct Bitmap {
    pub bitmap_indicator: u8,
    pub bmp: Vec<u8>,
}

#[derive(Debug)]
/// Data section
pub struct Data {
    pub data: Vec<f32>,
}

#[derive(Debug)]
/// Search parameters for when reading the file
pub struct SearchParams {
    pub param: u32,
    pub level: u32,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "json", derive(Serialize, Deserialize))]
/// Index information describing param, level, and where to find the sub-file
pub struct GribIndex {
    pub param: u8,
    pub level: u16,
    pub level_type: u8,
    pub offset: u64,
    pub length: u64,
}

impl Grib2Reader {
    /// Create a new instance of the GRIB1 reader by specifying the BufReader wrapping the file to read.
    pub fn new(buf_reader: BufReader<File>) -> Grib2Reader {
        Grib2Reader { reader: buf_reader, offset: 0 }
    }

    /// Read the file looking for data matching the specified search parameters and return the decoded result.
    pub async fn read(&mut self, search: Vec<SearchParams>) -> Result<Vec<Grib>, Grib2Error> {
        let mut offset = 0;
        let mut result = vec![];

        // We need to know how large the file is, so we know when to stop
        let length = self.reader.seek(SeekFrom::End(0)).await?;

        let mut count = 0;
        while offset < length {
            self.reader.seek(SeekFrom::Start(offset)).await?;

            let grib_result = self.read_grib(&search, true).await?;
            let length = match grib_result {
                GribResult::Grib(grib) => {
                    let length = grib.length;
                    result.push(grib);
                    length
                }
                GribResult::Length(length) => length,
            };
            count += 1;
            offset += length;
        }

        println!("File count: {count}");

        Ok(result)
    }

    // Keep calling to get next file from the container
    pub async fn read_binary_next(&mut self, file_length: u64) -> Result<Vec<u8>, Grib2Error> {
        if self.offset == file_length {
            return Ok(vec![]);
        }

        self.reader.seek(SeekFrom::Start(self.offset)).await?;

        let mut buffer = [0; 16];
        let _ = self.reader.read_exact(&mut buffer).await?;

        // Look for the letters GRIB that indicate this is indeed the kind of file we can read
        let header: [u8; 4] = [0x47, 0x52, 0x49, 0x42];
        if header != buffer[0..4] {
            return Err(Grib2Error::WrongHeader);
        }

        // Make sure this is indeed a version we can understand
        let version = buffer[7];
        if version != 2 {
            return Err(Grib2Error::WrongVersion(version));
        }

        // We use the length of the section to skip to the next one if we aren't interested in it
        let length_of_grib_section = read_u64_be(&buffer[8..]);

        self.reader.seek(SeekFrom::Start(self.offset)).await?;

        let mut data = vec![0; length_of_grib_section as usize];
        self.reader.read_exact(&mut data).await?;

        self.offset += length_of_grib_section;

        Ok(data)
    }

    async fn read_grib(&mut self, _search_list: &Vec<SearchParams>, _read_bds: bool) -> Result<GribResult, Grib2Error> {
        // The first 8 bytes describes the header of the grib1 file
        let mut buffer = [0; 16];
        let _ = self.reader.read_exact(&mut buffer).await?;

        // Look for the letters GRIB that indicate this is indeed the kind of file we can read
        let header: [u8; 4] = [0x47, 0x52, 0x49, 0x42];
        if header != buffer[0..4] {
            return Err(Grib2Error::WrongHeader);
        }

        // Make sure this is indeed a version we can understand
        let version = buffer[7];
        if version != 2 {
            return Err(Grib2Error::WrongVersion(version));
        }

        let discipline = buffer[6];

        // We use the length of the section to skip to the next one if we aren't interested in it
        let length_of_grib_section = read_u64_be(&buffer[8..]);

        let mut read_bytes = 16;

        let mut result_grib = Grib {
            length: length_of_grib_section,
            discipline: discipline,
            ..Default::default()
        };

        // Keep reading sections until we hit the end
        loop {
            let length = self.get_length().await?;

            read_bytes += length;

            let mut data = vec![0; length];
            self.reader.read_exact(&mut data).await?;

            let section_number = data[4];
            match section_number {
                1 => {
                    result_grib.identification = Some(self.parse_identification(&data));
                }
                3 => {
                    result_grib.grid_definition = Some(self.parse_grid_definition(&data));
                }
                4 => result_grib.product_definition.push(self.parse_product_definition(&data)),
                5 => result_grib.data_representation.push(self.parse_data_representation(&data)),
                6 => result_grib.bitmap.push(self.parse_bitmap(&data)),
                7 => result_grib.data.push(self.parse_data(&data, &result_grib.data_representation, &result_grib.bitmap)?),
                _ => {}
            }

            // Because the last section doesn't contain a length or a section number, we have to look at the length of the grib file,
            // and how much data we read to determine if we reached the end.
            // The last section has size 4, so if we are 4 bytes from the end, we must have hit the last section
            if read_bytes + 4 == length_of_grib_section as usize {
                break;
            }
        }

        Ok(GribResult::Grib(result_grib))
    }

    fn parse_identification(&self, buffer: &[u8]) -> Identification {
        Identification {
            identification_of_originating_generating_centre: read_u16_be(&buffer[5..]),
            identification_of_originating_generating_sub_centre: read_u16_be(&buffer[7..]),
            grib_master_tables_version_number: buffer[9],
            grib_local_tables_version_number: buffer[10],
            significance_of_reference_time: buffer[11],
            year: read_u16_be(&buffer[12..]),
            month: buffer[14],
            day: buffer[15],
            hour: buffer[16],
            minute: buffer[17],
            second: buffer[18],
            production_status_of_processed_data: buffer[19],
            type_of_processed_data: buffer[20],
        }
    }

    fn parse_grid_definition(&mut self, buffer: &[u8]) -> GridDefinition {
        let template = self.parse_grid_definition_template(buffer);

        GridDefinition {
            source_of_grid_definition: buffer[5],
            number_of_data_points: read_u32_be(&buffer[6..]),
            number_of_octets_for_optional_list_of_numbers_defining_number_of_points: buffer[10],
            interpretation_of_list_of_numbers_defining_number_of_points: buffer[11],
            grid_definition_template_number: read_u16_be(&buffer[12..]),
            template: template,
        }
    }

    fn parse_grid_definition_template(&self, buffer: &[u8]) -> GridDefinitionTemplate {
        let grid_definition_template_number = read_u16_be(&buffer[12..]);

        match grid_definition_template_number {
            30 => self.parse_lambert_conformal_template(buffer),
            _ => GridDefinitionTemplate::Unknown,
        }
    }

    fn parse_lambert_conformal_template(&self, buffer: &[u8]) -> GridDefinitionTemplate {
        GridDefinitionTemplate::LambertConformal(LambertConformalTemplate {
            shape_of_the_earth: buffer[14],
            scale_factor_of_radius_of_spherical_earth: buffer[15],
            scale_value_of_radius_of_spherical_earth: read_u32_be(&buffer[16..]),
            scale_factor_of_major_axis_of_oblate_spheroid_earth: buffer[20],
            scaled_value_of_major_axis_of_oblate_spheroid_earth: read_u32_be(&buffer[21..]),
            scale_factor_of_minor_axis_of_oblate_spheroid_earth: buffer[25],
            scaled_value_of_minor_axis_of_oblate_spheroid_earth: read_u32_be(&buffer[26..]),
            nx_number_of_points_along_the_x_axis: read_u32_be(&buffer[30..]),
            ny_number_of_points_along_the_y_axis: read_u32_be(&buffer[34..]),
            la1_latitude_of_first_grid_point: read_i32_be(&buffer[38..]),
            lo1_longitude_of_first_grid_point: read_i32_be(&buffer[42..]),
            resolution_and_component_flags: buffer[46],
            lad_latitude_where_dx_and_dy_are_specified: read_i32_be(&buffer[47..]),
            lov_longitude_of_meridian_parallel_to_y_axis_along_which_latitude_increases_as_the_y_coordinate_increases: read_i32_be(&buffer[51..]),
            dx_x_direction_grid_length: read_u32_be(&buffer[55..]),
            dy_y_direction_grid_length: read_u32_be(&buffer[59..]),
            projection_centre_flag: buffer[63],
            scanning_mode: buffer[64],
            latin_1_first_latitude_from_the_pole_at_which_the_secant_cone_cuts_the_sphere: read_i32_be(&buffer[65..]),
            latin_2_second_latitude_from_the_pole_at_which_the_secant_cone_cuts_the_sphere: read_i32_be(&buffer[69..]),
            latitude_of_the_southern_pole_of_projection: read_i32_be(&buffer[73..]),
            longitude_of_the_southern_pole_of_projection: read_i32_be(&buffer[77..]),
        })
    }

    fn parse_product_definition(&self, buffer: &[u8]) -> ProductDefinition {
        ProductDefinition {
            number_of_coordinate_values_after_template: read_u16_be(&buffer[5..]),
            product_definition_template_number: read_u16_be(&buffer[7..]),
            template: self.parse_product_definition_template(&buffer),
        }
    }

    fn parse_product_definition_template(&self, buffer: &[u8]) -> ProductDefinitionTemplate {
        let product_definition_template_number = read_u16_be(&buffer[7..]);

        match product_definition_template_number {
            1 => self.parse_id1_product_definition_template(buffer),
            11 => self.parse_id11_product_definition_template(buffer),
            _ => ProductDefinitionTemplate::Unknown,
        }
    }

    fn parse_id1_product_definition_template(&self, buffer: &[u8]) -> ProductDefinitionTemplate {
        ProductDefinitionTemplate::Id1(Id1ProductDefinitionTemplate {
            parameter_category: buffer[9],
            parameter_number: buffer[10],
            type_of_generating_process: buffer[11],
            background_generating_process_identifier_defined_by_originating_centre: buffer[12],
            forecast_generating_process_identified: buffer[13],
            hours_after_reference_time_data_cutoff: read_u16_be(&buffer[14..]),
            minutes_after_reference_time_data_cutoff: buffer[16],
            indicator_of_unit_of_time_range: buffer[17],
            forecast_time_in_units_defined_by_octet_18: read_u32_be(&buffer[18..]),
            type_of_first_fixed_surface: buffer[22],
            scale_factor_of_first_fixed_surface: buffer[23],
            scaled_value_of_first_fixed_surface: read_u32_be(&buffer[24..]),
            type_of_second_fixed_surfaced: buffer[28],
            scale_factor_of_second_fixed_surface: buffer[29],
            scaled_value_of_second_fixed_surfaces: read_u32_be(&buffer[30..]),
            type_of_ensemble_forecast: buffer[34],
            perturbation_number: buffer[35],
            number_of_forecasts_in_ensemble: buffer[36],
        })
    }

    fn parse_id11_product_definition_template(&self, buffer: &[u8]) -> ProductDefinitionTemplate {
        ProductDefinitionTemplate::Id11(Id11ProductDefinitionTemplate {
            parameter_category: buffer[9],
            parameter_number: buffer[10],
            type_of_generating_process: buffer[11],
            background_generating_process_identifier_defined_by_originating_centre: buffer[12],
            forecast_generating_process_identifier: buffer[13],
            hours_after_reference_time_data_cutoff: read_u16_be(&buffer[14..]),
            minutes_after_reference_time_data_cutoff: buffer[16],
            indicator_of_unit_of_time_range: buffer[17],
            forecast_time_in_units_defined_by_octet_18: read_u32_be(&buffer[18..]),
            type_of_first_fixed_surface: buffer[22],
            scale_factor_of_first_fixed_surface: buffer[23],
            scaled_value_of_first_fixed_surface: read_u32_be(&buffer[24..]),
            type_of_second_fixed_surfaced: buffer[28],
            scale_factor_of_second_fixed_surface: buffer[29],
            scaled_value_of_second_fixed_surfaces: read_u32_be(&buffer[30..]),
            type_of_ensemble_forecast: buffer[34],
            perturbation_number: buffer[35],
            number_of_forecasts_in_ensemble: buffer[36],
            year_of_end_of_overall_time_interval: read_u16_be(&buffer[37..]),
            month_of_end_of_overall_time_interval: buffer[39],
            day_of_end_of_overall_time_interval: buffer[40],
            hour_of_end_of_overall_time_interval: buffer[41],
            minute_of_end_overall_time_interval: buffer[42],
            second_of_end_of_overall_time_interval: buffer[43],
            n_number_of_time_ranges_specifications_describing_the_time_intervals_used_to_calculate_the_statistically_processed_field: buffer[44],
            total_number_of_data_values_missing_in_the_statistical_process: read_u32_be(&buffer[45..]),
        })
    }

    fn parse_data_representation(&self, buffer: &[u8]) -> DataRepresentation {
        DataRepresentation {
            number_of_data_points: read_u32_be(&buffer[5..]),
            data_representation_template_number: read_u16_be(&buffer[9..]),
            template: self.parse_data_representation_template(&buffer),
        }
    }

    fn parse_data_representation_template(&self, buffer: &[u8]) -> DataRepresentationTemplate {
        let data_representation_template_number = read_u16_be(&buffer[9..]);

        match data_representation_template_number {
            0 => self.parse_simple_packing_template(buffer),
            _ => DataRepresentationTemplate::Unknown,
        }
    }

    fn parse_simple_packing_template(&self, buffer: &[u8]) -> DataRepresentationTemplate {
        DataRepresentationTemplate::SimplePacking(SimplePackingTemplate {
            reference_value: read_f32_be(&buffer[11..]),
            binary_scale_factor: read_i16_be(&buffer[15..]),
            decimal_scale_factor: read_i16_be(&buffer[17..]),
            number_of_bits_used_for_each_packed_value: buffer[19],
            type_of_original_field_values: buffer[20],
        })
    }

    fn parse_bitmap(&self, buffer: &[u8]) -> Bitmap {
        Bitmap {
            bitmap_indicator: buffer[5],
            bmp: buffer[6..].to_vec(),
        }
    }

    fn parse_data(&self, buffer: &[u8], data_representation_list: &Vec<DataRepresentation>, bitmap: &Vec<Bitmap>) -> Result<Data, Grib2Error> {
        let mut r = BitReader::endian(Cursor::new(&buffer[5..]), BigEndian);

        let mut bitmap_reader = None;
        let uses_bitmap = bitmap[0].bitmap_indicator == 0;
        if uses_bitmap {
            bitmap_reader = Some(BitReader::endian(Cursor::new(&bitmap[0].bmp), BigEndian));
        }

        // We assume that the latest data representation is the use we need to use
        let cur_data_rep = &data_representation_list[data_representation_list.len() - 1];
        if let DataRepresentationTemplate::SimplePacking(sp) = &cur_data_rep.template {
            let number_of_data_points = cur_data_rep.number_of_data_points;

            let mut result: Vec<f32> = Vec::with_capacity(number_of_data_points as usize);

            let mut iterations = 0;
            let base: f32 = 2.0;
            let factor = base.powf(sp.binary_scale_factor as f32);

            while iterations < number_of_data_points {
                if uses_bitmap {
                    let present = match bitmap_reader.as_mut().unwrap().read_bit() {
                        Ok(val) => val,
                        Err(err) => {
                            return Err(Grib2Error::DataDecodeFailed(err.to_string()));
                        }
                    };

                    if !present {
                        result.push(0.0);
                        iterations += 1;
                        continue;
                    }
                }

                match r.read::<u32>(sp.number_of_bits_used_for_each_packed_value as u32) {
                    Ok(x) => {
                        let y = sp.reference_value + (x as f32) * factor;
                        result.push(y);
                    }
                    Err(err) => {
                        return Err(Grib2Error::DataDecodeFailed(format!("{:?}", err)));
                    }
                };
                iterations += 1;
            }

            return Ok(Data { data: result });
        }

        Err(Grib2Error::DataDecodeFailed("No SimplePacking info".into()))
    }

    async fn get_length(&mut self) -> Result<usize, Grib2Error> {
        // The header might be of variable length, so we read the length first, and then reset the position so the offsets in the documentation still fits
        let mut buffer = [0; 4];
        self.reader.read_exact(&mut buffer).await?;
        let len = read_u32_be(&buffer[..]) as usize;
        self.reader.seek(SeekFrom::Current(-4)).await?;

        Ok(len)
    }
}

//
// Utility functions to convert slices of memory into the value types we want
//

fn read_i16_be(array: &[u8]) -> i16 {
    let mut val = (array[1] as i16) + (((array[0] & 127) as i16) << 8);
    if array[0] & 0x80 > 0 {
        val = -val;
    }
    val
}

fn read_i32_be(array: &[u8]) -> i32 {
    let mut val = (array[3] as i32) + ((array[2] as i32) << 8) + ((array[1] as i32) << 16) + (((array[0] & 127) as i32) << 24);
    if array[0] & 0x80 > 0 {
        val = -val;
    }
    val
}

fn read_u16_be(array: &[u8]) -> u16 {
    (array[1] as u16) + ((array[0] as u16) << 8)
}

fn read_f32_be(array: &[u8]) -> f32 {
    let buf = [array[0], array[1], array[2], array[3]];
    f32::from_be_bytes(buf)
}

fn read_u32_be(array: &[u8]) -> u32 {
    (array[3] as u32) + ((array[2] as u32) << 8) + ((array[1] as u32) << 16) + ((array[0] as u32) << 24)
}

fn read_u64_be(array: &[u8]) -> u64 {
    (array[7] as u64) + ((array[6] as u64) << 8) + ((array[5] as u64) << 16) + ((array[4] as u64) << 24) + ((array[3] as u64) << 32) + ((array[2] as u64) << 40) + ((array[1] as u64) << 48) + ((array[0] as u64) << 56)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gif::{Encoder, Frame, Repeat};
    use proj4rs;
    use proj4rs::proj::Proj;
    use std::borrow::Cow;
    use std::f64::consts::PI;
    use tokio::io::AsyncWriteExt;
    pub const DEG_TO_RAD: f64 = PI / 180.0;
    pub const RAD_TO_DEG: f64 = 180.0 / PI;

    fn save_gif(result_data: &Vec<u8>, width: usize, height: usize, filename: &str) {
        let mut color_map = Vec::<u8>::with_capacity(256 * 3);

        // Add the palette
        for index in 0..255 {
            color_map.push(index);
            color_map.push(index);
            color_map.push(index);
        }

        color_map.push(255);
        color_map.push(255);
        color_map.push(255);

        let mut image = std::fs::File::create(filename).unwrap();
        let mut encoder = Encoder::new(&mut image, width as u16, height as u16, &color_map).unwrap();
        encoder.set_repeat(Repeat::Finite(0)).unwrap();

        let frame = Frame {
            width: width as u16,
            height: height as u16,
            buffer: Cow::Borrowed(&*result_data),
            ..Default::default()
        };

        encoder.write_frame(&frame).unwrap();
    }

    #[tokio::test]
    async fn read_test() -> Result<(), Grib2Error> {
        // cargo test --release read_test -- --nocapture > out.log
        let f = File::open("data/HARMONIE_DINI_SF_2024-03-21T120000Z_2024-03-21T140000Z.grib").await?;

        let mut reader = Grib2Reader::new(BufReader::new(f));
        let result = reader.read(vec![SearchParams { param: 33, level: 700 }, SearchParams { param: 34, level: 700 }]).await?;

        println!("Results:");
        for mut grib in result {
            grib.data[0].data = vec![];
            grib.bitmap[0].bmp = vec![];
            println!("{:#?}", &grib);
        }

        Ok(())
    }

    async fn get_data() -> Result<Vec<f32>, Grib2Error> {
        let f = File::open("data/HARMONIE_DINI_SF_5.grib").await?;

        let mut reader = Grib2Reader::new(BufReader::new(f));
        let result = reader.read(vec![SearchParams { param: 33, level: 700 }, SearchParams { param: 34, level: 700 }]).await?;

        let mut data: Vec<f32> = vec![];

        for grib in result {
            data = grib.data[0].data.clone();
        }

        Ok(data)
    }

    #[tokio::test]
    async fn read_single_test() -> Result<(), Grib2Error> {
        // cargo test --release read_single_test -- --nocapture > out_single.log
        let f = File::open("data/HARMONIE_DINI_SF_5.grib").await?;

        let mut reader = Grib2Reader::new(BufReader::new(f));
        let result = reader.read(vec![SearchParams { param: 33, level: 700 }, SearchParams { param: 34, level: 700 }]).await?;

        println!("Results:");
        for mut grib in result {
            let data = grib.data[0].data.clone();
            grib.data[0].data = vec![];

            println!("{:#?}", &grib);
            //println!("Data {:?}", &data);

            let maximum = *data.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
            let minimum = *data.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
            println!("Max value {:?}, min value: {:?}", maximum, minimum);
            let dx = 255. / (maximum - minimum);
            println!("Data dx: {:?}", dx);

            let img: Vec<u8> = data.iter().map(|v| f32::floor((v - minimum) * dx) as u8).collect();

            //nx_number_of_points_along_the_x_axis: 1906,
            //ny_number_of_points_along_the_y_axis: 1606,

            save_gif(&img, 1906, 1606, "./out/img.gif");
        }

        Ok(())
    }

    #[tokio::test]
    async fn mapview_test() -> Result<(), Grib2Error> {
        // cargo test --release mapview_test -- --nocapture > out.log

        // 800x600
        // UL 58.9238, 2.532 =>    281860.9506885687, 8163935.337341594
        // LR 52.3271, 18.1562 => 2021138.9387408334, 6859486.798858716

        // Max value 295.04468, min value: 245.35814
        let grib_data = get_data().await?;

        let from = Proj::from_proj_string("+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs").unwrap();
        let to = Proj::from_proj_string(concat!("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")).unwrap();
        //let to = Proj::from_proj_string("+proj=lcc +lat_0=55.5 +lon_0=-8 +lat_1=55.5 +lat_2=55.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs +type=crs").unwrap();

        let x_step = (2021138.9387408334 - 281860.9506885687) / 800.;
        let y_step = (8163935.337341594 - 6859486.798858716) / 600.;

        let x_start = 281860.9506885687; // x_min => +
        let y_start = 8163935.337341594; // y_max => -

        let lcc = hdf5_processor::LCC::new(55.5 * DEG_TO_RAD, -8. * DEG_TO_RAD, 55.5 * DEG_TO_RAD, 55.5 * DEG_TO_RAD);
        let (x_min, y_min) = lcc.forward(39.671000 * DEG_TO_RAD, (334.578000 - 360.) * DEG_TO_RAD);

        let dx = 2000.;
        let dy = 2000.;

        let nx = 1906 as usize;
        //let ny = 1606 as usize;

        let mut imm_data: Vec<f32> = vec![0.0; 800 * 600];

        let mut idx = 0;
        let mut y_cur = y_start;
        for _y in 0..600 {
            let mut x_cur = x_start;
            for _x in 0..800 {
                let mut point_3d = (x_cur, y_cur, 0.0);
                proj4rs::transform::transform(&from, &to, &mut point_3d).unwrap();

                let (x, y) = lcc.forward(point_3d.1, point_3d.0);

                let x_index = f32::round(((x - x_min) / dx as f64) as f32) as usize;
                let y_index = f32::round(((y - y_min) / dy as f64) as f32) as usize;
                let val = grib_data[y_index * nx + x_index];

                imm_data[idx] = val;
                idx += 1;

                //println!("wgs: {}, {} - lcc: {}, {}", px, py, x, y);

                x_cur += x_step;
            }
            y_cur -= y_step;
        }

        let max = 295.04468;
        let min = 245.35814;

        let dx = 255. / (max - min);
        println!("Data dx: {:?}", dx);

        let img: Vec<u8> = imm_data.iter().map(|v| f32::floor((v - min) * dx) as u8).collect();

        //nx_number_of_points_along_the_x_axis: 1906,
        //ny_number_of_points_along_the_y_axis: 1606,

        save_gif(&img, 800, 600, "./out/cut_img.gif");

        // point_3d.0 = point_3d.0.to_degrees();
        // point_3d.1 = point_3d.1.to_degrees();
        // println!("{} {}", point_3d.0, point_3d.1);

        Ok(())
    }

    #[tokio::test]
    async fn mapview_leae_test() -> Result<(), Grib2Error> {
        // cargo test --release mapview_leae_test -- --nocapture

        // 800x600
        // UL 58.9238, 2.532 =>    281860.9506885687, 8163935.337341594
        // LR 52.3271, 18.1562 => 2021138.9387408334, 6859486.798858716

        // x_max 352947.12941396347
        // x_min -222470.81951688128
        // y_max -3537939.7858127435
        // y_min -3937043.0019400464

        // {width: 1882, height: 1306}

        let width = 1882;
        let height = 1306;

        // Max value 295.04468, min value: 245.35814
        let grib_data = get_data().await?;

        let from = Proj::from_proj_string("+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs").unwrap();
        let to = Proj::from_proj_string(concat!("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")).unwrap();
        //let to = Proj::from_proj_string("+proj=lcc +lat_0=55.5 +lon_0=-8 +lat_1=55.5 +lat_2=55.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs +type=crs").unwrap();

        let x_step = (352947.12941396347 + 222470.81951688128) / width as f64;
        let y_step = (-3537939.7858127435 + 3937043.0019400464) / height as f64;

        let x_start = -222470.81951688128; // x_min => +
        let y_start = -3537939.7858127435; // y_max => -

        let laea = hdf5_processor::LAEA::new(90.0 * DEG_TO_RAD, 10.0 * DEG_TO_RAD);
        let lcc = hdf5_processor::LCC::new(55.5 * DEG_TO_RAD, -8. * DEG_TO_RAD, 55.5 * DEG_TO_RAD, 55.5 * DEG_TO_RAD);
        let (x_min, y_min) = lcc.forward(39.671000 * DEG_TO_RAD, (334.578000 - 360.) * DEG_TO_RAD);

        let dx = 2000.;
        let dy = 2000.;

        let nx = 1906 as usize;

        let mut imm_data: Vec<f32> = vec![0.0; width * height];

        let mut idx = 0;
        let mut y_cur = y_start;
        for _y in 0..height {
            let mut x_cur = x_start;
            for _x in 0..width {
                let (lon, lat) = laea.inverse(x_cur, y_cur);

                //let mut point_3d = (x_cur, y_cur, 0.0);
                //proj4rs::transform::transform(&from, &to, &mut point_3d).unwrap();

                let (x, y) = lcc.forward(lat, lon);

                let x_index = f32::round(((x - x_min) / dx as f64) as f32) as usize;
                let y_index = f32::round(((y - y_min) / dy as f64) as f32) as usize;
                let val = grib_data[y_index * nx + x_index];

                imm_data[idx] = val;
                idx += 1;

                //println!("wgs: {}, {} - lcc: {}, {}", px, py, x, y);

                x_cur += x_step;
            }
            y_cur -= y_step;
        }

        let max = 295.04468;
        let min = 245.35814;

        let dx = 255. / (max - min);
        println!("Data dx: {:?}", dx);

        let img: Vec<u8> = imm_data.iter().map(|v| f32::floor((v - min) * dx) as u8).collect();

        save_gif(&img, width, height, "./out/cut_img_laea.gif");

        Ok(())
    }

    #[tokio::test]
    async fn read_all_binary_test() -> Result<(), Grib2Error> {
        // cargo test --release read_all_binary_test -- --nocapture > out.log
        let f = File::open("data/HARMONIE_DINI_SF_2024-03-21T120000Z_2024-03-21T140000Z.grib").await?;

        let mut b_reader = BufReader::new(f);
        let file_length = b_reader.seek(SeekFrom::End(0)).await?;
        let mut reader = Grib2Reader::new(b_reader);

        let mut count = 0;
        loop {
            let _result = match reader.read_binary_next(file_length).await {
                Ok(data) => {
                    if data.is_empty() {
                        println!("All done");
                        break;
                    }
                    let mut file = File::create(format!("out/file{}.grib", &count)).await?;
                    file.write_all(&data).await?;
                    count += 1
                }
                Err(Grib2Error::EOF) => {
                    println!("EOF");
                    break;
                }
                Err(err) => {
                    println!("Err: {:?}", err);
                    break;
                }
            };
        }

        println!("File count: {}", count);

        Ok(())
    }
}
