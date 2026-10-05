import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { X, Camera } from 'lucide-react';

interface PersonListProps {
  label: string;
  persons: string[];
  onAdd: (name: string) => void;
  onRemove: (name: string) => void;
  images?: Record<string, string>; // this side's photos, by name
  onUploadImage?: (name: string, file: File) => void;
}

export function PersonList({ label, persons, onAdd, onRemove, images, onUploadImage }: PersonListProps) {
  const [name, setName] = useState('');
  const [warning, setWarning] = useState('');
  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  function handleAdd() {
    const trimmed = name.trim();
    if (!trimmed) return;
    if (persons.includes(trimmed)) {
      setWarning('Name already exists on this side');
      return;
    }
    onAdd(trimmed);
    setName('');
    setWarning('');
  }

  function handleFileChange(personName: string, e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file && onUploadImage) {
      onUploadImage(personName, file);
    }
    // Reset input so re-uploading the same file triggers change
    e.target.value = '';
  }

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setWarning('');
          }}
          onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
          placeholder={`Add ${label.toLowerCase().slice(0, -1)}...`}
          className="h-8 text-sm"
        />
        <Button size="sm" onClick={handleAdd} disabled={!name.trim()}>
          Add
        </Button>
      </div>
      {warning && <p className="text-xs text-red-500">{warning}</p>}
      <div className="flex flex-wrap gap-1">
        {persons.map((p) => (
          <span
            key={p}
            className="inline-flex items-center gap-1 rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium"
          >
            {images?.[p] && (
              <img
                src={images[p]}
                alt={p}
                className="w-4 h-4 rounded-full object-cover"
              />
            )}
            {p}
            {onUploadImage && (
              <>
                <button
                  onClick={() => fileInputRefs.current[p]?.click()}
                  className="text-muted-foreground hover:text-foreground"
                  aria-label={`Upload image for ${p}`}
                >
                  <Camera className="h-3 w-3" />
                </button>
                <input
                  ref={(el) => { fileInputRefs.current[p] = el; }}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => handleFileChange(p, e)}
                />
              </>
            )}
            <button
              onClick={() => onRemove(p)}
              className="text-muted-foreground hover:text-foreground"
              aria-label={`Remove ${p}`}
            >
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}
