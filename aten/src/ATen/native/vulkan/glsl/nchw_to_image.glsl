#version 450 core
#define PRECISION $precision
#define FORMAT    $format

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uImage;

/*
 * Input Buffer
 */
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
  float data[];
}
uBuffer;

/*
 * Params Buffer
 */
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // Extents of the output texture
  ivec4 out_extents;
  // Number of texels spanned by one channel
  ivec2 c_info;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uBlock.out_extents.xyz))) {
    return;
  }

  const int n_index = int(pos.z / uBlock.c_info.x);
  const int c_index = (pos.z % uBlock.c_info.x) * 4;
  int d_offset = (n_index * uBlock.c_info.y) + c_index;

  const int base_index =
      pos.x + uBlock.out_extents.x * pos.y + uBlock.out_extents.w * d_offset;
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * uBlock.out_extents.w;

  float val_x = 0;
  if (c_index < uBlock.c_info.y) {
    val_x = uBuffer.data[buf_indices.x];
  }
  float val_y = 0;
  if (c_index + 1 < uBlock.c_info.y) {
    val_y = uBuffer.data[buf_indices.y];
  }
  float val_z = 0;
  if (c_index + 2 < uBlock.c_info.y) {
    val_z = uBuffer.data[buf_indices.z];
  }
  float val_w = 0;
  if (c_index + 3 < uBlock.c_info.y) {
    val_w = uBuffer.data[buf_indices.w];
  }

  imageStore(uImage, pos, vec4(val_x, val_y, val_z, val_w));
}
