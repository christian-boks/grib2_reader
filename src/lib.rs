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
    pub la1_latitude_of_first_grid_point: u32,
    pub lo1_longitude_of_first_grid_point: u32,
    pub resolution_and_component_flags: u8,
    pub lad_latitude_where_dx_and_dy_are_specified: u32,
    pub lov_longitude_of_meridian_parallel_to_y_axis_along_which_latitude_increases_as_the_y_coordinate_increases: u32,
    pub dx_x_direction_grid_length: u32,
    pub dy_y_direction_grid_length: u32,
    pub projection_centre_flag: u8,
    pub scanning_mode: u8,
    pub latin_1_first_latitude_from_the_pole_at_which_the_secant_cone_cuts_the_sphere: u32,
    pub latin_2_second_latitude_from_the_pole_at_which_the_secant_cone_cuts_the_sphere: u32,
    pub latitude_of_the_southern_pole_of_projection: u32,
    pub longitude_of_the_southern_pole_of_projection: u32,
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

        println!("before read exact");
        let mut buffer = [0; 16];
        let _ = self.reader.read_exact(&mut buffer).await?;
        println!("read exact");

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
                7 => result_grib.data.push(self.parse_data(&data, &result_grib.data_representation)?),
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
            la1_latitude_of_first_grid_point: read_u32_be(&buffer[38..]),
            lo1_longitude_of_first_grid_point: read_u32_be(&buffer[42..]),
            resolution_and_component_flags: buffer[46],
            lad_latitude_where_dx_and_dy_are_specified: read_u32_be(&buffer[47..]),
            lov_longitude_of_meridian_parallel_to_y_axis_along_which_latitude_increases_as_the_y_coordinate_increases: read_u32_be(&buffer[51..]),
            dx_x_direction_grid_length: read_u32_be(&buffer[55..]),
            dy_y_direction_grid_length: read_u32_be(&buffer[59..]),
            projection_centre_flag: buffer[63],
            scanning_mode: buffer[64],
            latin_1_first_latitude_from_the_pole_at_which_the_secant_cone_cuts_the_sphere: read_u32_be(&buffer[65..]),
            latin_2_second_latitude_from_the_pole_at_which_the_secant_cone_cuts_the_sphere: read_u32_be(&buffer[69..]),
            latitude_of_the_southern_pole_of_projection: read_u32_be(&buffer[73..]),
            longitude_of_the_southern_pole_of_projection: read_u32_be(&buffer[77..]),
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

    fn parse_data(&self, buffer: &[u8], data_representation_list: &Vec<DataRepresentation>) -> Result<Data, Grib2Error> {
        let mut r = BitReader::endian(Cursor::new(&buffer[5..]), BigEndian);

        let mut result = vec![];

        // We assume that the latest data representation is the use we need to use
        let cur_data_rep = &data_representation_list[data_representation_list.len() - 1];
        if let DataRepresentationTemplate::SimplePacking(sp) = &cur_data_rep.template {
            let number_of_data_points = cur_data_rep.number_of_data_points;

            let mut iterations = 0;
            let base: f32 = 2.0;
            let factor = base.powf(sp.binary_scale_factor as f32);

            while iterations < number_of_data_points {
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
        }

        Ok(Data { data: result })
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
    use tokio::io::AsyncWriteExt;

    use super::*;

    #[tokio::test]
    async fn read_test() -> Result<(), Grib2Error> {
        // cargo test --release read_test -- --nocapture > out.log
        let f = File::open("data/HARMONIE_DINI_SF_2024-03-21T120000Z_2024-03-21T140000Z.grib").await?;

        let mut reader = Grib2Reader::new(BufReader::new(f));
        let result = reader.read(vec![SearchParams { param: 33, level: 700 }, SearchParams { param: 34, level: 700 }]).await?;

        //assert_eq!(2, result.len());

        // assert_eq!(result[0].pds.indicator_of_parameter_and_units, 33);
        // assert_eq!(result[0].pds.level_or_layer_value, 700);

        // assert_eq!(result[1].pds.indicator_of_parameter_and_units, 34);
        // assert_eq!(result[1].pds.level_or_layer_value, 700);

        // println!("Results:");
        // for grib in result {
        //     println!("{:#?}", &grib.pds);
        //     if let Some(gds) = grib.gds {
        //         println!("{:#?}", &gds);
        //     }
        // }

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
