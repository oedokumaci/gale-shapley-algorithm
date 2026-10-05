import { useState, useCallback } from 'react';

import type { PersonImages, Side } from '@/types';

export function usePersonImages(initialImages: PersonImages = { proposers: {}, responders: {} }) {
  const [images, setImages] = useState<PersonImages>(initialImages);

  const uploadImage = useCallback((side: Side, name: string, file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setImages((prev) => ({ ...prev, [side]: { ...prev[side], [name]: reader.result as string } }));
      }
    };
    reader.readAsDataURL(file);
  }, []);

  const removeImage = useCallback((side: Side, name: string) => {
    setImages((prev) => {
      const updated = { ...prev[side] };
      delete updated[name];
      return { ...prev, [side]: updated };
    });
  }, []);

  return { images, uploadImage, removeImage };
}
