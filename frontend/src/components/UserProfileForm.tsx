import { useState, useEffect, FormEvent, ChangeEvent } from 'react';
import { UserProfile } from '@/lib/api/types';
import {
  validateAge,
  validateHobbies,
  validateBudget,
  validateRequired,
  formatBudget,
} from '@/lib/utils/validation';

interface UserProfileFormProps {
  onSubmit: (profile: UserProfile) => void;
  isLoading?: boolean;
}

interface FormErrors {
  age?: string;
  hobbies?: string;
  relationship?: string;
  budget?: string;
  occasion?: string;
  personalityTraits?: string;
}

const HOBBY_OPTIONS = [
  'Spor',
  'Müzik',
  'Okuma',
  'Seyahat',
  'Yemek Pişirme',
  'Bahçecilik',
  'Fotoğrafçılık',
  'Resim',
  'Teknoloji',
  'El Sanatları',
  'Oyun',
  'Sinema',
];

const RELATIONSHIP_OPTIONS = [
  'Anne',
  'Baba',
  'Eş',
  'Sevgili',
  'Kardeş',
  'Arkadaş',
  'İş Arkadaşı',
  'Diğer',
];

const OCCASION_OPTIONS = [
  'Doğum Günü',
  'Yıldönümü',
  'Sevgililer Günü',
  'Anneler Günü',
  'Babalar Günü',
  'Yılbaşı',
  'Mezuniyet',
  'Diğer',
];

const PERSONALITY_TRAIT_OPTIONS = [
  'Pratik',
  'Romantik',
  'Sportif',
  'Entelektüel',
  'Sanatsal',
  'Teknoloji Meraklısı',
  'Doğa Sever',
  'Sosyal',
];

export function UserProfileForm({ onSubmit, isLoading = false }: UserProfileFormProps) {
  const [formData, setFormData] = useState<UserProfile>({
    age: 0,
    hobbies: [],
    relationship: '',
    budget: 0,
    occasion: '',
    personalityTraits: [],
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [budgetDisplay, setBudgetDisplay] = useState<string>('');

  // Real-time validation
  useEffect(() => {
    const newErrors: FormErrors = {};

    if (touched.age) {
      const ageValidation = validateAge(formData.age);
      if (!ageValidation.isValid) {
        newErrors.age = ageValidation.error;
      }
    }

    if (touched.hobbies) {
      const hobbiesValidation = validateHobbies(formData.hobbies);
      if (!hobbiesValidation.isValid) {
        newErrors.hobbies = hobbiesValidation.error;
      }
    }

    if (touched.relationship) {
      const relationshipValidation = validateRequired(formData.relationship, 'İlişki durumu');
      if (!relationshipValidation.isValid) {
        newErrors.relationship = relationshipValidation.error;
      }
    }

    if (touched.budget) {
      const budgetValidation = validateBudget(formData.budget);
      if (!budgetValidation.isValid) {
        newErrors.budget = budgetValidation.error;
      }
    }

    if (touched.occasion) {
      const occasionValidation = validateRequired(formData.occasion, 'Özel gün');
      if (!occasionValidation.isValid) {
        newErrors.occasion = occasionValidation.error;
      }
    }

    setErrors(newErrors);
  }, [formData, touched]);

  const handleAgeChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setFormData({ ...formData, age: value === '' ? 0 : parseFloat(value) });
    setTouched({ ...touched, age: true });
  };

  const handleHobbyToggle = (hobby: string) => {
    const newHobbies = formData.hobbies.includes(hobby)
      ? formData.hobbies.filter((h) => h !== hobby)
      : [...formData.hobbies, hobby];
    
    setFormData({ ...formData, hobbies: newHobbies });
    setTouched({ ...touched, hobbies: true });
  };

  const handleRelationshipChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setFormData({ ...formData, relationship: e.target.value });
    setTouched({ ...touched, relationship: true });
  };

  const handleBudgetChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const numValue = value === '' ? 0 : parseFloat(value);
    
    setFormData({ ...formData, budget: numValue });
    setBudgetDisplay(value);
    setTouched({ ...touched, budget: true });
  };

  const handleBudgetBlur = () => {
    if (formData.budget > 0) {
      setBudgetDisplay(formatBudget(formData.budget));
    }
  };

  const handleBudgetFocus = () => {
    setBudgetDisplay(formData.budget > 0 ? formData.budget.toString() : '');
  };

  const handleOccasionChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setFormData({ ...formData, occasion: e.target.value });
    setTouched({ ...touched, occasion: true });
  };

  const handlePersonalityTraitToggle = (trait: string) => {
    const newTraits = formData.personalityTraits.includes(trait)
      ? formData.personalityTraits.filter((t) => t !== trait)
      : [...formData.personalityTraits, trait];
    
    if (newTraits.length <= 5) {
      setFormData({ ...formData, personalityTraits: newTraits });
    }
  };

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};
    let isValid = true;

    const ageValidation = validateAge(formData.age);
    if (!ageValidation.isValid) {
      newErrors.age = ageValidation.error;
      isValid = false;
    }

    const hobbiesValidation = validateHobbies(formData.hobbies);
    if (!hobbiesValidation.isValid) {
      newErrors.hobbies = hobbiesValidation.error;
      isValid = false;
    }

    const relationshipValidation = validateRequired(formData.relationship, 'İlişki durumu');
    if (!relationshipValidation.isValid) {
      newErrors.relationship = relationshipValidation.error;
      isValid = false;
    }

    const budgetValidation = validateBudget(formData.budget);
    if (!budgetValidation.isValid) {
      newErrors.budget = budgetValidation.error;
      isValid = false;
    }

    const occasionValidation = validateRequired(formData.occasion, 'Özel gün');
    if (!occasionValidation.isValid) {
      newErrors.occasion = occasionValidation.error;
      isValid = false;
    }

    setErrors(newErrors);
    setTouched({
      age: true,
      hobbies: true,
      relationship: true,
      budget: true,
      occasion: true,
    });

    return isValid;
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (validateForm()) {
      onSubmit(formData);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-2xl mx-auto p-4 sm:p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md">
      <h2 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6 text-gray-900 dark:text-white">
        Hediye Alacağınız Kişinin Bilgileri
      </h2>

      {/* Age Input */}
      <div className="mb-4 sm:mb-6">
        <label htmlFor="age" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Yaş *
        </label>
        <input
          type="number"
          id="age"
          value={formData.age || ''}
          onChange={handleAgeChange}
          className={`w-full px-4 py-3 sm:py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white text-base touch-manipulation ${
            errors.age ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
          }`}
          placeholder="18-100 arası bir değer girin"
          min="18"
          max="100"
          disabled={isLoading}
        />
        {errors.age && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.age}</p>
        )}
      </div>

      {/* Hobbies Multi-Select */}
      <div className="mb-4 sm:mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Hobiler * (En az 1, en fazla 10)
        </label>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {HOBBY_OPTIONS.map((hobby) => (
            <button
              key={hobby}
              type="button"
              onClick={() => handleHobbyToggle(hobby)}
              disabled={isLoading}
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors touch-target touch-manipulation ${
                formData.hobbies.includes(hobby)
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200 active:bg-gray-300 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600 dark:active:bg-gray-500'
              }`}
            >
              {hobby}
            </button>
          ))}
        </div>
        {errors.hobbies && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.hobbies}</p>
        )}
      </div>

      {/* Relationship Select */}
      <div className="mb-4 sm:mb-6">
        <label htmlFor="relationship" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          İlişki Durumu *
        </label>
        <select
          id="relationship"
          value={formData.relationship}
          onChange={handleRelationshipChange}
          disabled={isLoading}
          className={`w-full px-4 py-3 sm:py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white text-base touch-manipulation ${
            errors.relationship ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
          }`}
        >
          <option value="">Seçiniz</option>
          {RELATIONSHIP_OPTIONS.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        {errors.relationship && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.relationship}</p>
        )}
      </div>

      {/* Budget Input */}
      <div className="mb-4 sm:mb-6">
        <label htmlFor="budget" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Bütçe (TL) *
        </label>
        <input
          type="text"
          id="budget"
          value={budgetDisplay}
          onChange={handleBudgetChange}
          onFocus={handleBudgetFocus}
          onBlur={handleBudgetBlur}
          disabled={isLoading}
          className={`w-full px-4 py-3 sm:py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white text-base touch-manipulation ${
            errors.budget ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
          }`}
          placeholder="Örn: 500"
        />
        {errors.budget && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.budget}</p>
        )}
      </div>

      {/* Occasion Select */}
      <div className="mb-4 sm:mb-6">
        <label htmlFor="occasion" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Özel Gün *
        </label>
        <select
          id="occasion"
          value={formData.occasion}
          onChange={handleOccasionChange}
          disabled={isLoading}
          className={`w-full px-4 py-3 sm:py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white text-base touch-manipulation ${
            errors.occasion ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
          }`}
        >
          <option value="">Seçiniz</option>
          {OCCASION_OPTIONS.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        {errors.occasion && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.occasion}</p>
        )}
      </div>

      {/* Personality Traits Multi-Select */}
      <div className="mb-4 sm:mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Kişilik Özellikleri (İsteğe bağlı, en fazla 5)
        </label>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {PERSONALITY_TRAIT_OPTIONS.map((trait) => (
            <button
              key={trait}
              type="button"
              onClick={() => handlePersonalityTraitToggle(trait)}
              disabled={isLoading || (formData.personalityTraits.length >= 5 && !formData.personalityTraits.includes(trait))}
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors touch-target touch-manipulation ${
                formData.personalityTraits.includes(trait)
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200 active:bg-gray-300 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600 dark:active:bg-gray-500'
              } ${
                formData.personalityTraits.length >= 5 && !formData.personalityTraits.includes(trait)
                  ? 'opacity-50 cursor-not-allowed'
                  : ''
              }`}
            >
              {trait}
            </button>
          ))}
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-medium hover:bg-blue-700 active:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors touch-target touch-manipulation"
      >
        {isLoading ? 'Yükleniyor...' : 'Hediye Önerisi Al'}
      </button>
    </form>
  );
}
