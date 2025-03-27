"""Model architectures for inpainting detection."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

def build_cnn_model(image_size=256, learning_rate=1e-4):
    """Build CNN model with skip connections for inpainting detection."""
    inputs = layers.Input(shape=(image_size, image_size, 3))
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    block1_output = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(block1_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    block2_output = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(block2_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    block3_output = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 4 with skip connection from Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(block3_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Prepare block2 output for skip connection (match dimensions)
    skip_connection = layers.Conv2D(128, (1, 1), padding='same')(block2_output)
    skip_connection = layers.MaxPooling2D(pool_size=(2, 2))(skip_connection)  # Match spatial dimensions
    
    # Add skip connection
    x = layers.Add()([x, skip_connection])
    x = layers.Activation('relu')(x)  # Activation after adding
    block4_output = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 5 with skip connection from Block 3
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(block4_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Prepare block3 output for skip connection
    skip_connection2 = layers.Conv2D(256, (1, 1), padding='same')(block3_output)
    skip_connection2 = layers.MaxPooling2D(pool_size=(2, 2))(skip_connection2)  # Match spatial dimensions
    
    # Add skip connection
    x = layers.Add()([x, skip_connection2])
    x = layers.Activation('relu')(x)  # Activation after adding
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Final layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    return model, None

def build_resnet_model(image_size=256, learning_rate=1e-4):

    # Base ResNet50 model without top layers
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3)
    )
    
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(2e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(2e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile the model
    model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    return model, base_model

def build_efficientnetv2_model(image_size=256, learning_rate=1e-4):
    """Build EfficientNetV2 model for inpainting detection."""
    # Base EfficientNetV2S model without top layers
    base_model = keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3)
    )
    
    # Initially freeze all layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # First dense block
    x = layers.Dense(512, activation=None, kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Second dense block
    x = layers.Dense(256, activation=None, kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile the model
    model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    return model, base_model