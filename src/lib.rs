#![feature(doc_auto_cfg)]
//! This is designed to do two things:
//! 1. Cut up a combined grib2 file into smaller individual grib2 parts using tokio and async.
//! 2. Parse a single grib2 file from a `Vec<u8>` blob (without tokio and async).
//!

use bitstream_io::{BigEndian, BitRead, BitReader};
use error::Grib2Error;

use std::io::Cursor;
#[cfg(feature = "async")]
use std::io::SeekFrom;
#[cfg(feature = "async")]
use tokio::io::{AsyncReadExt, AsyncSeekExt, BufReader};
pub mod error;

/// The star of the show when doing async
#[cfg(feature = "async")]
pub struct Grib2Reader<R> {
    pub reader: BufReader<R>,
    offset: u64,
}

/// The star of the show when only parsing
pub struct Grib2Parser {
    buffer: Vec<u8>,
    index: usize,
}

#[derive(Debug, Default)]
/// Grib2 file representation
pub struct Grib2 {
    pub length: u64,
    pub discipline: u8,
    pub identification: Option<Identification>,
    pub grid_definition: Option<GridDefinition>,
    pub product_definition: Vec<ProductDefinition>,
    pub data_representation: Vec<DataRepresentation>,
    pub bitmap: Vec<Bitmap>,
    pub data: Vec<Data>,
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
/// Lambert Conformal Template
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
/// Grid Definition Template
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

impl ProductDefinition {
    /// Get the parameter category from a product definition
    pub fn get_parameter_category(&self) -> u8 {
        match &self.template {
            ProductDefinitionTemplate::Id1(pdt) => pdt.parameter_category,
            ProductDefinitionTemplate::Id11(pdt) => pdt.parameter_category,
            _ => 255,
        }
    }

    /// Get the parameter number from a product definition
    pub fn get_parameter_number(&self) -> u8 {
        match &self.template {
            ProductDefinitionTemplate::Id1(pdt) => pdt.parameter_number,
            ProductDefinitionTemplate::Id11(pdt) => pdt.parameter_number,
            _ => 255,
        }
    }
}

#[derive(Debug, Clone)]
/// Product Definition Template
pub enum ProductDefinitionTemplate {
    Id1(Id1ProductDefinitionTemplate),
    Id11(Id11ProductDefinitionTemplate),
    Unknown,
}

#[derive(Debug, Clone, Default)]
/// Id1 Product Definition Template
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
/// Id11 Product Definition Template
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
/// Data Representation
pub struct DataRepresentation {
    pub number_of_data_points: u32,
    pub data_representation_template_number: u16,
    pub template: DataRepresentationTemplate,
}

#[derive(Debug, Clone)]
/// Data Representation Template
pub enum DataRepresentationTemplate {
    SimplePacking(SimplePackingTemplate),
    Unknown,
}

#[derive(Debug, Clone)]
/// Simple Packing Template
pub struct SimplePackingTemplate {
    reference_value: f32,
    binary_scale_factor: i16,
    #[allow(dead_code)]
    decimal_scale_factor: i16,
    number_of_bits_used_for_each_packed_value: u8,
    #[allow(dead_code)]
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

#[cfg(feature = "async")]
impl<R> Grib2Reader<R>
where
    R: AsyncReadExt,
    R: AsyncSeekExt,
    R: Unpin,
{
    /// Create a new instance of the GRIB2 reader by specifying the BufReader wrapping the file to read.
    pub fn new(buf_reader: BufReader<R>) -> Grib2Reader<R> {
        Grib2Reader { reader: buf_reader, offset: 0 }
    }

    /// Read the file and return all the decoded results.
    pub async fn read(&mut self) -> Result<Vec<Grib2>, Grib2Error> {
        let mut offset = 0;
        let mut result = vec![];

        // We need to know how large the file is, so we know when to stop
        let length = self.reader.seek(SeekFrom::End(0)).await?;

        while offset < length {
            self.reader.seek(SeekFrom::Start(offset)).await?;

            let grib_result = self.read_grib().await?;
            offset += grib_result.length;
            result.push(grib_result);
        }

        Ok(result)
    }

    /// Keep calling to get next file from the container
    pub async fn read_binary_next(&mut self, file_length: u64) -> Result<Vec<u8>, Grib2Error> {
        if self.offset == file_length {
            return Ok(vec![]);
        }

        self.reader.seek(SeekFrom::Start(self.offset)).await?;

        let mut buffer = [0; 16];
        let _ = self.reader.read_exact(&mut buffer).await?;

        let length_of_grib_section = check_header_and_get_length(&buffer)?;

        self.reader.seek(SeekFrom::Start(self.offset)).await?;

        let mut data = vec![0; length_of_grib_section as usize];
        self.reader.read_exact(&mut data).await?;

        self.offset += length_of_grib_section;

        Ok(data)
    }

    async fn read_grib(&mut self) -> Result<Grib2, Grib2Error> {
        // The first 8 bytes describes the header of the grib file
        let mut buffer = [0; 16];
        let _ = self.reader.read_exact(&mut buffer).await?;

        let length_of_grib_section = check_header_and_get_length(&buffer)?;

        let mut read_bytes = 16;

        let mut result_grib = Grib2 {
            length: length_of_grib_section,
            discipline: buffer[6],
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
                    result_grib.identification = Some(parse_identification(&data));
                }
                3 => {
                    result_grib.grid_definition = Some(parse_grid_definition(&data));
                }
                4 => result_grib.product_definition.push(parse_product_definition(&data)),
                5 => result_grib.data_representation.push(parse_data_representation(&data)),
                6 => result_grib.bitmap.push(parse_bitmap(&data)),
                7 => result_grib.data.push(parse_data(&data, &result_grib.data_representation, &result_grib.bitmap)?),
                _ => {}
            }

            // Because the last section doesn't contain a length or a section number, we have to look at the length of the grib file,
            // and how much data we read to determine if we reached the end.
            // The last section has size 4, so if we are 4 bytes from the end, we must have hit the last section
            if read_bytes + 4 == length_of_grib_section as usize {
                break;
            }
        }

        Ok(result_grib)
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

impl Default for Grib2Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Grib2Parser {
    /// Create a new instance of the GRIB2 reader.
    pub fn new() -> Grib2Parser {
        Grib2Parser { buffer: vec![], index: 0 }
    }

    /// Parse the passed in buffer and return any found grib information
    pub fn parse(&mut self, buffer: Vec<u8>) -> Result<Grib2, Grib2Error> {
        self.buffer = buffer;
        self.index = 0;

        let buffer: &[u8] = &self.buffer[self.index..self.index + 16];
        self.index += 16;

        let length_of_grib_section = check_header_and_get_length(buffer)?;

        let mut read_bytes = 16;

        let mut result_grib = Grib2 {
            length: length_of_grib_section,
            discipline: buffer[6],
            ..Default::default()
        };

        // Keep reading sections until we hit the end
        loop {
            let length = self.get_length();
            read_bytes += length;

            let data: &[u8] = &self.buffer[self.index..self.index + length];
            self.index += length;

            let section_number = data[4];
            match section_number {
                1 => {
                    result_grib.identification = Some(parse_identification(data));
                }
                3 => {
                    result_grib.grid_definition = Some(parse_grid_definition(data));
                }
                4 => result_grib.product_definition.push(parse_product_definition(data)),
                5 => result_grib.data_representation.push(parse_data_representation(data)),
                6 => result_grib.bitmap.push(parse_bitmap(data)),
                7 => result_grib.data.push(parse_data(data, &result_grib.data_representation, &result_grib.bitmap)?),
                _ => {}
            }

            // Because the last section doesn't contain a length or a section number, we have to look at the length of the grib file,
            // and how much data we read to determine if we reached the end.
            // The last section has size 4, so if we are 4 bytes from the end, we must have hit the last section
            if read_bytes + 4 == length_of_grib_section as usize {
                break;
            }
        }

        Ok(result_grib)
    }

    fn get_length(&mut self) -> usize {
        let buffer: &[u8] = &self.buffer[self.index..self.index + 4];
        read_u32_be(buffer) as usize
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

fn check_header_and_get_length(buffer: &[u8]) -> Result<u64, Grib2Error> {
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

    Ok(length_of_grib_section)
}

fn parse_identification(buffer: &[u8]) -> Identification {
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

fn parse_grid_definition(buffer: &[u8]) -> GridDefinition {
    let template = parse_grid_definition_template(buffer);

    GridDefinition {
        source_of_grid_definition: buffer[5],
        number_of_data_points: read_u32_be(&buffer[6..]),
        number_of_octets_for_optional_list_of_numbers_defining_number_of_points: buffer[10],
        interpretation_of_list_of_numbers_defining_number_of_points: buffer[11],
        grid_definition_template_number: read_u16_be(&buffer[12..]),
        template,
    }
}

fn parse_grid_definition_template(buffer: &[u8]) -> GridDefinitionTemplate {
    let grid_definition_template_number = read_u16_be(&buffer[12..]);

    match grid_definition_template_number {
        30 => parse_lambert_conformal_template(buffer),
        _ => GridDefinitionTemplate::Unknown,
    }
}

fn parse_lambert_conformal_template(buffer: &[u8]) -> GridDefinitionTemplate {
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

fn parse_product_definition(buffer: &[u8]) -> ProductDefinition {
    ProductDefinition {
        number_of_coordinate_values_after_template: read_u16_be(&buffer[5..]),
        product_definition_template_number: read_u16_be(&buffer[7..]),
        template: parse_product_definition_template(buffer),
    }
}

fn parse_product_definition_template(buffer: &[u8]) -> ProductDefinitionTemplate {
    let product_definition_template_number = read_u16_be(&buffer[7..]);

    match product_definition_template_number {
        1 => parse_id1_product_definition_template(buffer),
        11 => parse_id11_product_definition_template(buffer),
        _ => ProductDefinitionTemplate::Unknown,
    }
}

fn parse_id1_product_definition_template(buffer: &[u8]) -> ProductDefinitionTemplate {
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

fn parse_id11_product_definition_template(buffer: &[u8]) -> ProductDefinitionTemplate {
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

fn parse_data_representation(buffer: &[u8]) -> DataRepresentation {
    DataRepresentation {
        number_of_data_points: read_u32_be(&buffer[5..]),
        data_representation_template_number: read_u16_be(&buffer[9..]),
        template: parse_data_representation_template(buffer),
    }
}

fn parse_data_representation_template(buffer: &[u8]) -> DataRepresentationTemplate {
    let data_representation_template_number = read_u16_be(&buffer[9..]);

    match data_representation_template_number {
        0 => parse_simple_packing_template(buffer),
        _ => DataRepresentationTemplate::Unknown,
    }
}

fn parse_simple_packing_template(buffer: &[u8]) -> DataRepresentationTemplate {
    DataRepresentationTemplate::SimplePacking(SimplePackingTemplate {
        reference_value: read_f32_be(&buffer[11..]),
        binary_scale_factor: read_i16_be(&buffer[15..]),
        decimal_scale_factor: read_i16_be(&buffer[17..]),
        number_of_bits_used_for_each_packed_value: buffer[19],
        type_of_original_field_values: buffer[20],
    })
}

fn parse_bitmap(buffer: &[u8]) -> Bitmap {
    Bitmap {
        bitmap_indicator: buffer[5],
        bmp: buffer[6..].to_vec(),
    }
}

fn parse_data(buffer: &[u8], data_representation_list: &[DataRepresentation], bitmap: &[Bitmap]) -> Result<Data, Grib2Error> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "async"))]
    use std::{fs::File, io::Read};

    #[cfg(feature = "async")]
    use tokio::fs::File;

    #[test]
    #[cfg(not(feature = "async"))]
    fn read_single_test() {
        let mut f = File::open("data/HARMONIE_DINI_SF_5.grib").expect("Unable to open file");

        let mut data = vec![];
        f.read_to_end(&mut data).expect("Unable to read file");

        let mut grib2_parser = Grib2Parser::new();
        let mut grib = grib2_parser.parse(data).expect("Unable to parse grib2 file");

        println!("Results:");
        // We don't want to display the binary data, so remove that from the output
        grib.data[0].data = vec![];
        println!("{:#?}", &grib);
    }

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn read_single_test_async() -> Result<(), Grib2Error> {
        let f = File::open("data/HARMONIE_DINI_SF_5.grib").await?;

        let mut reader = Grib2Reader::new(BufReader::new(f));
        let result = reader.read().await?;

        println!("Results:");
        for mut grib in result {
            grib.data[0].data = vec![];
            println!("{:#?}", &grib);
        }

        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "async")]
    async fn read_all_binary_test() -> Result<(), Grib2Error> {
        let f = File::open("data/HARMONIE_DINI_SF_5.grib").await?;

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
